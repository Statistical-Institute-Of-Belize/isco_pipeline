import os
import pandas as pd
import numpy as np
import re
import logging
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from transformers import RobertaModel, RobertaTokenizer
import torch

from .utils import load_config, ensure_dir, get_device

# Configure logging
logger = logging.getLogger(__name__)

def clean_and_combine_text(job_titles, duties_descriptions):
    """
    Clean and combine job titles and duties descriptions
    
    Args:
        job_titles (Series): Series of job titles
        duties_descriptions (Series): Series of duties descriptions
        
    Returns:
        Series: Combined and cleaned text
    """
    # Convert to string type in case they're numeric
    job_titles = job_titles.astype(str)
    
    # Fill missing values with placeholders
    job_titles = job_titles.fillna("no title")
    duties_descriptions = duties_descriptions.fillna("no description")
    
    # Remove rows with empty strings after strip
    job_titles = job_titles.str.strip()
    duties_descriptions = duties_descriptions.str.strip()
    
    # Replace completely empty strings with placeholders
    job_titles = job_titles.replace('', 'no title')
    duties_descriptions = duties_descriptions.replace('', 'no description')
    
    # Combine with separator
    combined = job_titles + " | " + duties_descriptions
    
    # Clean by lowercasing and removing non-word characters except "|"
    cleaned = combined.str.lower().str.replace(r"[^\w\s|]", "", regex=True)
    
    # Final check for empty strings
    cleaned = cleaned.replace('', 'unknown occupation')
    
    return cleaned

def load_valid_isco_codes(config):
    """
    Load valid ISCO codes from the reference file
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        set: Set of valid ISCO codes
    """
    valid_codes = set()
    
    try:
        # Get reference file path from config or use default
        reference_path = config["data"].get(
            "reference_file", 
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data", "reference", "isco08_reference.csv"
            )
        )
        
        # Convert relative path to absolute if needed
        if not os.path.isabs(reference_path):
            reference_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                reference_path
            )
        
        if not os.path.exists(reference_path):
            logger.warning(f"ISCO-08 reference file not found at {reference_path}")
            return valid_codes
        
        # Read the reference file with multiple encoding attempts
        isco_df = None
        encodings = ['utf-8', 'latin-1', 'windows-1252', 'ISO-8859-1']
        
        for encoding in encodings:
            try:
                isco_df = pd.read_csv(reference_path, encoding=encoding)
                logger.info(f"Successfully loaded reference file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                logger.debug(f"Failed to decode with {encoding} encoding, trying next...")
                if encoding == encodings[-1]:  # Last encoding attempt
                    logger.warning(f"Failed to decode file with any encoding")
            except Exception as e:
                logger.warning(f"Error reading ISCO reference file: {e}")
                return valid_codes
        
        # If all encoding attempts failed
        if isco_df is None:
            logger.warning("Could not read ISCO reference file with any encoding")
            return valid_codes
        
        # Check if the expected column exists
        if 'ISCO 08 Code' not in isco_df.columns:
            logger.warning("ISCO 08 Code column not found in reference file")
            return valid_codes
        
        # Extract codes and validate
        for code in isco_df['ISCO 08 Code'].dropna().unique():
            code_str = str(code).strip()
            
            # Skip non-numeric codes and keep valid digits
            if code_str and code_str.isdigit():
                valid_codes.add(code_str)
                
                # Handle special case for armed forces codes (may be stored as integers without leading zeros)
                if len(code_str) == 3 and code_str in ['110', '210', '310']:
                    armed_forces_code = '0' + code_str
                    valid_codes.add(armed_forces_code)
                    logger.debug(f"Added armed forces code with leading zero: {armed_forces_code}")
        
        logger.info(f"Loaded {len(valid_codes)} valid ISCO codes from reference file")
        
        # Load parent codes (1-3 digit) if they're not already in the reference
        # This ensures all valid hierarchy levels are included
        all_codes = set(valid_codes)
        for code in list(valid_codes):
            if len(code) == 4:  # For 4-digit codes, add 1, 2, and 3 digit parents
                all_codes.add(code[0])        # Major group (1-digit)
                all_codes.add(code[:2])       # Sub-major group (2-digit)
                all_codes.add(code[:3])       # Minor group (3-digit)
            elif len(code) == 3:  # For 3-digit codes, add 1 and 2 digit parents
                all_codes.add(code[0])        # Major group (1-digit)
                all_codes.add(code[:2])       # Sub-major group (2-digit)
            elif len(code) == 2:  # For 2-digit codes, add 1-digit parent
                all_codes.add(code[0])        # Major group (1-digit)
        
        # Update with all hierarchical codes
        if len(all_codes) > len(valid_codes):
            logger.info(f"Added {len(all_codes) - len(valid_codes)} parent codes to valid codes list")
            valid_codes = all_codes
        
    except Exception as e:
        logger.warning(f"Error loading ISCO reference file: {e}")
    
    return valid_codes

def validate_isco_codes(df, output_dir, config):
    """
    Ensure ISCO codes are valid and exist in the reference file if configured
    
    Args:
        df (DataFrame): DataFrame with ISCO codes
        output_dir (str): Directory to save invalid codes
        config (dict): Configuration dictionary
        
    Returns:
        DataFrame: DataFrame with valid ISCO codes
    """
    # Check if ISCO code is present
    if "isco_code" not in df.columns:
        raise ValueError("DataFrame must contain 'isco_code' column")
    
    # Convert to string if not already
    df["isco_code"] = df["isco_code"].astype(str)
    
    # Initial record count for reporting
    initial_count = len(df)
    
    # Get config setting for reference validation
    validate_against_reference = config["data"].get("validate_against_reference", True)
    
    # Initial validation - check for numeric codes
    # Allow 1-4 digit codes (all levels of ISCO hierarchy)
    format_valid_mask = df["isco_code"].str.match(r"^\d{1,4}$")
    
    # Handle armed forces special case - they may be stored as integers without leading zeros
    # For example, '110' should be '0110', '210' should be '0210', '310' should be '0310'
    if 'isco_code' in df.columns:
        # Create a temporary copy of isco_codes to update armed forces codes with leading zeros
        armed_forces_mask = df["isco_code"].isin(['110', '210', '310'])
        if armed_forces_mask.any():
            logger.info(f"Found {armed_forces_mask.sum()} potential armed forces codes stored as integers without leading zeros")
            # Add leading zero to armed forces codes
            df.loc[armed_forces_mask, "isco_code"] = '0' + df.loc[armed_forces_mask, "isco_code"]
            logger.info("Added leading zeros to armed forces codes")
    
    # Further validation - check if code exists in reference
    if validate_against_reference:
        # Load valid codes from reference file
        valid_reference_codes = load_valid_isco_codes(config)
        
        if valid_reference_codes:
            logger.info("Validating ISCO codes against reference file")
            reference_valid_mask = df["isco_code"].isin(valid_reference_codes)
            
            # Track codes that are format-valid but not in reference
            unknown_codes = df[format_valid_mask & ~reference_valid_mask]["isco_code"].unique()
            if len(unknown_codes) > 0:
                logger.warning(f"Found {len(unknown_codes)} ISCO codes that are not in the reference file")
                logger.info(f"Example unknown codes: {', '.join(unknown_codes[:min(10, len(unknown_codes))])}")
                
                # Add a more prominent warning if we're filtering these out
                if len(unknown_codes) > len(df) * 0.05:  # If more than 5% will be removed
                    from colorama import Fore, Style
                    logger.warning(f"{Fore.YELLOW}{Style.BRIGHT}⚠️ WARNING: {len(unknown_codes)} unique ISCO codes "
                                   f"({len(df[format_valid_mask & ~reference_valid_mask])} records) "
                                   f"will be removed because they don't exist in the reference file.{Style.RESET_ALL}")
                    logger.warning(f"{Fore.YELLOW}If you want to keep these records, set validate_against_reference: false "
                                   f"in config.yaml{Style.RESET_ALL}")
            
            # Combined validation - code must be both format-valid and in reference
            valid_mask = format_valid_mask & reference_valid_mask
        else:
            logger.warning("No reference codes loaded, falling back to format validation only")
            valid_mask = format_valid_mask
    else:
        # Only perform format validation
        logger.info("Reference validation disabled, using format validation only")
        valid_mask = format_valid_mask
    
    # Save invalid records
    invalid_df = df[~valid_mask].copy()
    if len(invalid_df) > 0:
        ensure_dir(output_dir)
        invalid_df.to_csv(os.path.join(output_dir, "invalid_codes.csv"), index=False)
        logger.warning(f"Found {len(invalid_df)} records with invalid ISCO codes")
    
    # Return only valid records
    valid_df = df[valid_mask].copy()
    filtered_count = initial_count - len(valid_df)
    
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} records with invalid ISCO codes ({filtered_count/initial_count:.1%} of data)")
    
    logger.info(f"Kept {len(valid_df)} records with valid ISCO codes (from {initial_count} total)")
    
    return valid_df

def _prepare_texts_for_embeddings(texts):
    """Normalize incoming texts and drop empty entries."""

    normalized = [str(text) for text in texts]
    filtered = [text for text in normalized if text and text.strip()]

    if len(filtered) < len(normalized):
        logger.debug(
            "Filtered %d empty or whitespace-only texts before embedding generation",
            len(normalized) - len(filtered),
        )

    return filtered


def _iter_chunks(items, size):
    for index in range(0, len(items), size):
        yield items[index : index + size], index // size + 1


def _build_embedding_matrix(tokenizer, model, texts, chunk_size, batch_size, device):
    from tqdm import tqdm
    from colorama import Fore, Style

    embeddings = []
    total_chunks = max(1, (len(texts) - 1) // chunk_size + 1)

    progress_bar = tqdm(
        total=total_chunks,
        desc=f"{Fore.BLUE}Generating embeddings{Style.RESET_ALL}",
        unit="chunk",
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    with torch.no_grad():
        for chunk, chunk_number in _iter_chunks(texts, chunk_size):
            logger.debug("Processing embeddings for chunk %d/%d", chunk_number, total_chunks)
            chunk_embeddings = []

            for batch_start in range(0, len(chunk), batch_size):
                batch = chunk[batch_start : batch_start + batch_size]
                batch = [str(text).strip() for text in batch]
                batch = [text if text else "unknown" for text in batch]

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

                inputs = None
                outputs = None

                try:
                    inputs = tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=128,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    outputs = model(**inputs)
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    chunk_embeddings.append(batch_embeddings)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("Error processing batch: %s", exc)
                    logger.error("Problematic batch: %s", batch)
                finally:
                    if inputs is not None:
                        del inputs
                    if outputs is not None:
                        del outputs

            if chunk_embeddings:
                try:
                    embeddings.append(np.vstack(chunk_embeddings))
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error("Error combining chunk embeddings: %s", exc)

            progress_bar.update(1)
            progress_bar.set_postfix(current=f"Chunk {chunk_number}/{total_chunks}")

            import gc

            gc.collect()

    progress_bar.close()

    if not embeddings:
        return None

    try:
        from colorama import Fore, Style

        logger.info(
            f"{Fore.BLUE}Combining embeddings from {len(embeddings)} chunks...{Style.RESET_ALL}"
        )
        matrix = np.vstack(embeddings)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Error combining embeddings: %s", exc)
        return None

    if matrix.size == 0 or np.isnan(matrix).any():
        logger.warning("Invalid embeddings detected. Returning default clusters.")
        return None

    from colorama import Fore, Style

    logger.info(
        f"{Fore.GREEN}Successfully created embedding matrix of shape {matrix.shape}{Style.RESET_ALL}"
    )
    return matrix


def _cluster_large_embeddings(embeddings):
    from sklearn.neighbors import NearestNeighbors
    from tqdm import tqdm
    from colorama import Fore, Style

    logger.info(
        f"{Fore.BLUE}Large embedding matrix detected. Using memory-efficient clustering.{Style.RESET_ALL}"
    )

    sample_size = min(50000, len(embeddings) // 2 or len(embeddings))
    logger.info(
        f"{Fore.BLUE}Sampling {sample_size:,} of {len(embeddings):,} embeddings for initial clustering...{Style.RESET_ALL}"
    )

    indices = np.random.choice(len(embeddings), sample_size, replace=False)
    sampled_embeddings = embeddings[indices]

    cluster_progress = tqdm(
        total=100,
        desc=f"{Fore.BLUE}DBSCAN clustering{Style.RESET_ALL}",
        bar_format="{l_bar}{bar}| {elapsed}<{remaining}",
        ncols=80,
    )

    clustering = DBSCAN(
        eps=0.7,
        min_samples=2,
        metric="cosine",
        n_jobs=-1,
        algorithm="ball_tree",
    ).fit(sampled_embeddings)

    cluster_progress.n = cluster_progress.total
    cluster_progress.close()

    core_sample_indices = indices[clustering.core_sample_indices_]
    core_samples = embeddings[core_sample_indices]
    core_labels = clustering.labels_[clustering.core_sample_indices_]

    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(core_samples)

    all_labels = np.full(len(embeddings), -1)
    all_labels[core_sample_indices] = core_labels

    remaining_indices = np.setdiff1d(np.arange(len(embeddings)), core_sample_indices)
    batch_size = 1000

    for start in range(0, len(remaining_indices), batch_size):
        batch_indices = remaining_indices[start : start + batch_size]
        batch_embeddings = embeddings[batch_indices]
        distances, neighbours = nn.kneighbors(batch_embeddings)

        for position, (dist, neighbour_idx) in enumerate(zip(distances, neighbours)):
            if dist[0] <= 0.7:  # Same eps threshold as DBSCAN
                all_labels[batch_indices[position]] = core_labels[neighbour_idx[0]]

    return all_labels


def _cluster_embedding_matrix(embeddings):
    if len(embeddings) > 50000:
        return _cluster_large_embeddings(embeddings)

    logger.info("Clustering embeddings with standard DBSCAN")
    clustering = DBSCAN(
        eps=0.6,
        min_samples=3,
        metric="cosine",
        n_jobs=-1,
    ).fit(embeddings)

    return clustering.labels_


def cluster_embeddings(texts, model_name, chunk_size=64, batch_size=8):
    """
    Cluster texts using embeddings to identify outliers
    
    Args:
        texts (list): List of text strings
        model_name (str): Name of the model to use for embeddings
        chunk_size (int): Chunk size for processing
        batch_size (int): Batch size for processing each chunk
        
    Returns:
        np.ndarray: Cluster labels
    """
    device = get_device()
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name).to(device)
    model.eval()

    texts = _prepare_texts_for_embeddings(texts)
    if not texts:
        logger.warning("No valid texts found for clustering. Returning default clusters.")
        return np.zeros(1)

    embedding_matrix = _build_embedding_matrix(tokenizer, model, texts, chunk_size, batch_size, device)
    if embedding_matrix is None:
        logger.warning("No valid embeddings generated. Returning default clusters.")
        return np.zeros(len(texts))

    try:
        labels = _cluster_embedding_matrix(embedding_matrix)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Error during clustering: %s", exc)
        logger.warning("Returning default clusters due to error.")
        return np.zeros(len(texts))

    if len(labels) != len(texts):
        logger.warning(
            "Number of cluster labels (%d) doesn't match number of texts (%d). Padding with zeros.",
            len(labels),
            len(texts),
        )
        padded_labels = np.zeros(len(texts))
        padded_labels[: len(labels)] = labels
        return padded_labels

    return labels

def save_outliers(df, output_dir):
    """
    Identify and save clusters with more than one unique ISCO code
    
    Args:
        df (DataFrame): DataFrame with cluster labels
        output_dir (str): Directory to save outliers
    """
    # Find clusters with multiple ISCO codes
    outliers = df.groupby("cluster").filter(lambda x: x["isco_code"].nunique() > 1)
    
    if len(outliers) > 0:
        ensure_dir(output_dir)
        outliers.to_csv(os.path.join(output_dir, "outliers.csv"), index=False)
        logger.info(f"Found {len(outliers)} potential outliers across {outliers['cluster'].nunique()} clusters")

def split_and_save_data(df, output_dir):
    """
    Split data into train, validation, and test sets
    
    Args:
        df (DataFrame): DataFrame to split
        output_dir (str): Directory to save splits
    """
    # First check for rare ISCO codes that might cause stratification issues
    code_counts = df['isco_code'].value_counts()
    rare_codes = code_counts[code_counts <= 2]
    if len(rare_codes) > 0:
        logger.info(f"Found {len(rare_codes)} ISCO codes with ≤2 examples (may cause stratification issues)")
        # If we have a lot of rare codes, log some examples
        if len(rare_codes) > 20:
            sample_rare = rare_codes.head(10).index.tolist()
            logger.debug(f"Examples of rare codes: {', '.join(map(str, sample_rare))}")
        else:
            logger.debug(f"Rare codes: {', '.join(map(str, rare_codes.index.tolist()))}")
    
    try:
        # Try to split with stratification by ISCO code
        train_df, temp_df = train_test_split(
            df, test_size=0.3, random_state=42, stratify=df["isco_code"]
        )
        
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=42, stratify=temp_df["isco_code"]
        )
        
        logger.info("Successfully performed stratified split by ISCO code")
    except ValueError as e:
        # Fall back to random split if stratification fails (e.g., with small datasets)
        logger.warning(f"Stratified split failed: {e}. Falling back to random split.")
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Save each split with text and isco_code columns
    train_df[["text", "isco_code"]].to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df[["text", "isco_code"]].to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df[["text", "isco_code"]].to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    logger.info(f"Split data into {len(train_df)} train, {len(val_df)} validation, and {len(test_df)} test samples")

def preprocess_data(input_csv, config):
    """
    Preprocess raw CSV data
    
    Args:
        input_csv (str): Path to input CSV
        config (dict): Configuration dictionary
    """
    # Load CSV with required columns
    try:
        df = pd.read_csv(input_csv)
        required_columns = ["job_title", "duties_description", "isco_code"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
    except Exception as e:
        logger.error(f"Error loading input CSV: {e}")
        raise
    
    # Filter out records with invalid ISCO codes (DK/NS/missing)
    initial_count = len(df)
    
    # Filter out common missing/DK/NS codes
    invalid_codes = ["999999"]
    mask = ~df["isco_code"].astype(str).isin(invalid_codes)
    df = df[mask]
    
    # Filter out too long codes (exceeding 4 digits)
    str_codes = df["isco_code"].astype(str)
    mask = str_codes.str.len() <= 4
    df = df[mask]
    
    # Filter out non-numeric codes
    mask = str_codes.str.match(r'^\d+$')
    df = df[mask]
    
    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} records with invalid ISCO codes (missing, non-standard, or non-numeric)")
    
    # Clean and combine text
    logger.info("Cleaning and combining text")
    df["text"] = clean_and_combine_text(df["job_title"], df["duties_description"])
    
    # Validate ISCO codes
    logger.info("Validating ISCO codes")
    df = validate_isco_codes(df, config["data"]["processed_dir"], config)
    
    # Check config settings for dataset size handling
    skip_clustering = config["data"].get("skip_clustering", False)
    skip_threshold = config["data"].get("skip_clustering_threshold", 100000)
    
    # Skip clustering if explicitly set in config or if dataset is very large
    if skip_clustering or len(df) > skip_threshold:
        logger.warning(f"Skipping clustering step (dataset has {len(df)} rows, threshold is {skip_threshold})")
        # Assign all examples to cluster 0 (skip clustering)
        df["cluster"] = 0
        
        # Save outliers
        logger.info("Skipping outlier detection due to dataset size")
        
        # Split and save data
        logger.info("Splitting and saving data")
        split_and_save_data(df, config["data"]["processed_dir"])
        
        return
        
    # Check if dataset is large - for large datasets, we need special handling
    if len(df) > 50000:
        logger.warning(f"Large dataset detected ({len(df)} rows).")
        
        # For moderately large datasets, ask if user wants to proceed with full dataset or use a sample
        try:
            proceed = input(f"Dataset has {len(df)} rows which may cause memory issues during clustering. Proceed with full dataset? (y/n): ")
            if proceed.lower() != 'y':
                sample_size = min(10000, len(df))
                logger.info(f"Using a random sample of {sample_size} rows")
                df = df.sample(sample_size, random_state=42)
        except:
            # In non-interactive environments, automatically use a sample
            sample_size = min(10000, len(df))
            logger.info(f"Non-interactive mode: using a random sample of {sample_size} rows")
            df = df.sample(sample_size, random_state=42)
    
    # Try to use CPU for clustering to avoid OOM on MPS
    import os
    old_force_cpu = os.environ.get("FORCE_CPU", "0")
    try:
        # Force CPU for clustering step
        os.environ["FORCE_CPU"] = "1"
        logger.info("Temporarily switching to CPU for embedding generation to avoid OOM errors")
        
        # Cluster texts using embeddings
        logger.info("Clustering texts using embeddings")
        chunk_size = min(64, len(df))  # Ensure chunk size doesn't exceed dataset size
        batch_size = min(8, chunk_size)  # Ensure batch size doesn't exceed chunk size
        df["cluster"] = cluster_embeddings(
            df["text"].tolist(),
            config["model"]["name"],
            chunk_size=chunk_size,
            batch_size=batch_size
        )
    finally:
        # Restore original setting
        os.environ["FORCE_CPU"] = old_force_cpu
    
    # Save outliers
    logger.info("Identifying and saving outliers")
    save_outliers(df, config["data"]["processed_dir"])
    
    # Split and save data
    logger.info("Splitting and saving data")
    split_and_save_data(df, config["data"]["processed_dir"])

if __name__ == "__main__":
    # Load config
    config = load_config()
    
    # Preprocess data
    preprocess_data("/path/to/input.csv", config)
