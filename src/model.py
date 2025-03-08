import os
import pandas as pd
import json
import logging
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)

from .utils import (
    load_config, 
    ensure_dir, 
    get_device, 
    MPSAutocast, 
    MPSGradScaler, 
    configure_mps_memory
)

# Configure logging
logger = logging.getLogger(__name__)

class MPSOptimizedTrainer(Trainer):
    """
    Custom Trainer class with MPS-specific optimizations for mixed precision training.
    
    This trainer extends the Hugging Face Trainer to add support for mixed precision
    training on Apple Silicon (M1/M2/M3) with MPS backend, similar to how 
    fp16 training works on CUDA devices.
    
    It's designed to be a drop-in replacement for the standard Trainer.
    """
    def __init__(self, *args, **kwargs):
        # Get mixed precision settings from config
        self.config = kwargs.get("args").training_config
        
        # MPS specific settings
        self.use_mps = torch.backends.mps.is_available()
        self.use_mp = self.config.get("mps_mixed_precision", True) if self.use_mps else False
        
        # Create gradient scaler for mixed precision
        self.scaler = MPSGradScaler(
            enabled=self.use_mp,
            init_scale=self.config.get("mp_init_scale", 128.0),
            growth_factor=self.config.get("mp_growth_factor", 2.0),
            backoff_factor=self.config.get("mp_backoff_factor", 0.5),
            growth_interval=self.config.get("mp_growth_interval", 2000)
        )
        
        # Initialize parent class
        super().__init__(*args, **kwargs)
        
        # Log mixed precision status
        if self.use_mp:
            logger.info("MPS mixed precision training enabled")
        elif self.use_mps:
            logger.info("MPS detected but mixed precision training disabled")
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step with mixed precision support for MPS.
        
        Args:
            model: Model to train
            inputs: Inputs for the model
            num_items_in_batch: Optional parameter for gradient accumulation and batch size adjustment
            
        Returns:
            torch.Tensor: Loss
        """
        model.train()
        
        # Move inputs to device
        inputs = self._prepare_inputs(inputs)
        
        # Mixed precision context manager
        with MPSAutocast(enabled=self.use_mp):
            loss = self.compute_loss(model, inputs)
            
        if self.args.gradient_accumulation_steps > 1 and num_items_in_batch is not None:
            # In newer Transformers versions, they pass this parameter for better gradient accumulation
            if num_items_in_batch != inputs["labels"].shape[0]:
                # Only adjust for partial batches
                loss = loss * (num_items_in_batch / inputs["labels"].shape[0])
            loss = loss / self.args.gradient_accumulation_steps
            
        # Scale loss and run backward pass
        if self.use_mp:
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
        else:
            loss.backward()
            
        return loss.detach()
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval, metrics=None, logs=None):
        """Log, save and evaluate as in parent class but handle MPS-specific ops"""
        # Same as parent with minor modifications for MPS
        result = super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval, metrics, logs)
        return result
    
    def optimizer_step(self, closure=None):
        """
        Optimizer step with mixed precision support for MPS.
        
        Args:
            closure: Closure for optimizer
        """
        # Skip step if no optimization required
        if self.args.gradient_accumulation_steps > 1:
            if self.state.global_step % self.args.gradient_accumulation_steps != 0:
                return
                
        # Apply gradient clipping if needed
        if self.args.max_grad_norm > 0:
            if self.use_mp:
                # First unscale gradients
                self.scaler.unscale_(self.optimizer)
                # Then clip as usual
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                
        # Step optimizer with scaler if using mixed precision
        if self.use_mp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            
        self.optimizer.zero_grad()

class ISCODataset(Dataset):
    """
    Dataset class for ISCO code classification
    """
    def __init__(self, texts, labels, tokenizer, max_seq_length):
        """
        Initialize dataset
        
        Args:
            texts (list): List of text strings
            labels (list): List of label indices
            tokenizer: Tokenizer to use
            max_seq_length (int): Maximum sequence length
        """
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding='max_length',
            max_length=max_seq_length,
            return_tensors="pt"
        )
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get an item from the dataset
        
        Args:
            idx (int): Index
            
        Returns:
            dict: Item with encodings and label
        """
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        """
        Get dataset length
        
        Returns:
            int: Dataset length
        """
        return len(self.labels)

def get_all_valid_isco_codes():
    """
    Get all valid ISCO-08 codes from the official reference file
    
    Returns:
        list: List of all valid ISCO-08 codes as strings
    """
    try:
        # Load the official ISCO-08 reference file
        reference_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     "data", "reference", "isco08_reference.csv")
        
        if not os.path.exists(reference_path):
            logger.warning(f"ISCO-08 reference file not found at {reference_path}. Using fallback code generation.")
            return generate_fallback_isco_codes()
            
        # Read the reference file and extract codes
        isco_df = pd.read_csv(reference_path)
        
        # Check if the expected column exists
        if 'ISCO 08 Code' not in isco_df.columns:
            logger.warning("ISCO 08 Code column not found in reference file. Using fallback code generation.")
            return generate_fallback_isco_codes()
            
        # Extract all codes from the reference file
        valid_codes = [str(code).strip() for code in isco_df['ISCO 08 Code'].dropna().unique()]
        
        # Filter out non-numeric codes and empty strings
        valid_codes = [code for code in valid_codes if code and code.isdigit()]
        
        logger.info(f"Loaded {len(valid_codes)} valid ISCO-08 codes from official reference file")
        return valid_codes
        
    except Exception as e:
        logger.warning(f"Error loading ISCO-08 reference file: {e}. Using fallback code generation.")
        return generate_fallback_isco_codes()

def generate_fallback_isco_codes():
    """
    Generate ISCO-08 codes as a fallback when the reference file is not available
    Using a more accurate approximation than generating all combinations
    
    Returns:
        list: List of generated ISCO-08 codes as strings
    """
    # Define major groups (1-digit)
    major_groups = [str(i) for i in range(0, 10)]  # 0-9
    
    # Define sub-major groups (2-digit)
    submajor_groups = []
    # For each major group
    for major in range(0, 10):
        # Armed Forces (major group 0)
        if major == 0:
            submajor_groups.extend([f"{major}{i}" for i in range(1, 4)])  # 01-03
        # Regular major groups 1-9
        elif major > 0:
            # Most major groups have 1-9 sub-major groups
            max_submajor = 7 if major in [6, 8] else 9  # 6 and 8 have fewer
            submajor_groups.extend([f"{major}{i}" for i in range(1, max_submajor + 1)])
    
    # Define minor groups (3-digit) and unit groups (4-digit)
    minor_groups = []
    unit_groups = []
    
    # Armed Forces (major group 0)
    minor_groups.extend(["011", "021", "031"])
    unit_groups.extend(["0110", "0210", "0310"])
    
    # Major group 1: Managers
    for sub in ["11", "12", "13", "14"]:
        for i in range(1, 5 if sub == "14" else 10):
            minor = f"{sub}{i}"
            minor_groups.append(minor)
            for j in range(1, 5):  # Most unit groups have 1-4 subdivisions
                unit = f"{minor}{j}"
                if int(unit) <= 1439:  # ISCO-08 doesn't go beyond 1439
                    unit_groups.append(unit)
    
    # Major group 2: Professionals
    for sub in ["21", "22", "23", "24", "25", "26"]:
        for i in range(1, 10):
            minor = f"{sub}{i}"
            if int(minor) <= 265:  # ISCO-08 doesn't go beyond 265
                minor_groups.append(minor)
                for j in range(1, 5):  # Most unit groups have 1-4 subdivisions
                    unit = f"{minor}{j}"
                    if int(unit) <= 2656:  # ISCO-08 doesn't go beyond 2656
                        unit_groups.append(unit)
    
    # Major group 3: Technicians and Associate Professionals
    for sub in ["31", "32", "33", "34", "35"]:
        for i in range(1, 10):
            minor = f"{sub}{i}"
            if int(minor) <= 352:  # ISCO-08 doesn't go beyond 352
                minor_groups.append(minor)
                for j in range(1, 5):  # Most unit groups have 1-4 subdivisions
                    unit = f"{minor}{j}"
                    if int(unit) <= 3522:  # ISCO-08 doesn't go beyond 3522
                        unit_groups.append(unit)
    
    # Major group 4: Clerical Support Workers
    for sub in ["41", "42", "43", "44"]:
        for i in range(1, 5):
            minor = f"{sub}{i}"
            if int(minor) <= 441:  # ISCO-08 doesn't go beyond 441
                minor_groups.append(minor)
                for j in range(1, 5):  # Most unit groups have 1-4 subdivisions
                    unit = f"{minor}{j}"
                    if int(unit) <= 4419:  # ISCO-08 doesn't go beyond 4419
                        unit_groups.append(unit)
    
    # Major group 5: Service and Sales Workers
    for sub in ["51", "52", "53", "54"]:
        for i in range(1, 5):
            minor = f"{sub}{i}"
            if int(minor) <= 541:  # ISCO-08 doesn't go beyond 541
                minor_groups.append(minor)
                for j in range(1, 5):  # Most unit groups have 1-4 subdivisions
                    unit = f"{minor}{j}"
                    if int(unit) <= 5419:  # ISCO-08 doesn't go beyond 5419
                        unit_groups.append(unit)
    
    # Major group 6: Skilled Agricultural, Forestry and Fishery Workers
    for sub in ["61", "62", "63"]:
        for i in range(1, 5):
            minor = f"{sub}{i}"
            if int(minor) <= 634:  # ISCO-08 doesn't go beyond 634
                minor_groups.append(minor)
                for j in range(1, 5):  # Most unit groups have 1-4 subdivisions
                    unit = f"{minor}{j}"
                    if int(unit) <= 6340:  # ISCO-08 doesn't go beyond 6340
                        unit_groups.append(unit)
    
    # Major group 7: Craft and Related Trades Workers
    for sub in ["71", "72", "73", "74", "75"]:
        for i in range(1, 6):
            minor = f"{sub}{i}"
            if int(minor) <= 754:  # ISCO-08 doesn't go beyond 754
                minor_groups.append(minor)
                for j in range(1, 5):  # Most unit groups have 1-4 subdivisions
                    unit = f"{minor}{j}"
                    if int(unit) <= 7549:  # ISCO-08 doesn't go beyond 7549
                        unit_groups.append(unit)
    
    # Major group 8: Plant and Machine Operators and Assemblers
    for sub in ["81", "82", "83"]:
        for i in range(1, 5):
            minor = f"{sub}{i}"
            if int(minor) <= 835:  # ISCO-08 doesn't go beyond 835
                minor_groups.append(minor)
                for j in range(1, 5):  # Most unit groups have 1-4 subdivisions
                    unit = f"{minor}{j}"
                    if int(unit) <= 8350:  # ISCO-08 doesn't go beyond 8350
                        unit_groups.append(unit)
    
    # Major group 9: Elementary Occupations
    for sub in ["91", "92", "93", "94", "95", "96"]:
        for i in range(1, 5):
            minor = f"{sub}{i}"
            if int(minor) <= 962:  # ISCO-08 doesn't go beyond 962
                minor_groups.append(minor)
                for j in range(1, 5):  # Most unit groups have 1-4 subdivisions
                    unit = f"{minor}{j}"
                    if int(unit) <= 9629:  # ISCO-08 doesn't go beyond 9629
                        unit_groups.append(unit)
    
    # Combine all codes
    all_codes = major_groups + submajor_groups + minor_groups + unit_groups
    
    # Remove duplicates and sort
    unique_codes = sorted(list(set(all_codes)))
    
    logger.info(f"Generated {len(unique_codes)} approximate ISCO-08 codes as fallback")
    
    return unique_codes

def load_isco_reference(config=None):
    """
    Load ISCO-08 reference data from the official reference file
    
    Args:
        config (dict, optional): Configuration dictionary
        
    Returns:
        tuple: 
            - DataFrame with reference data
            - Set of valid ISCO codes
            - Dict mapping codes to titles
    """
    valid_codes = set()
    code_to_title = {}
    
    try:
        # Get reference path from config if available
        if config and "data" in config and "reference_file" in config["data"]:
            reference_path = config["data"]["reference_file"]
            # Convert relative path to absolute if needed
            if not os.path.isabs(reference_path):
                reference_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    reference_path
                )
        else:
            # Fallback to default path
            reference_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "data", "reference", "isco08_reference.csv"
            )
        
        if not os.path.exists(reference_path):
            logger.warning(f"ISCO-08 reference file not found at {reference_path}")
            return None, valid_codes, code_to_title
            
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
                return None, valid_codes, code_to_title
        
        # If all encoding attempts failed
        if isco_df is None:
            logger.warning("Could not read ISCO reference file with any encoding")
            return None, valid_codes, code_to_title
        
        # Check if the expected columns exist
        required_cols = ['ISCO 08 Code', 'Title EN']
        if not all(col in isco_df.columns for col in required_cols):
            logger.warning(f"Reference file is missing required columns: {required_cols}")
            return None, valid_codes, code_to_title
            
        # Extract all codes and titles
        for _, row in isco_df.iterrows():
            try:
                code = str(row['ISCO 08 Code']).strip()
                title = str(row['Title EN']).strip()
                
                # Skip empty or non-numeric codes
                if not code or not code.isdigit():
                    continue
                    
                # Add to valid codes set
                valid_codes.add(code)
                
                # Add to code_to_title mapping
                code_to_title[code] = title
            except Exception as e:
                logger.debug(f"Error processing ISCO reference row: {e}")
        
        # Also add parent codes (1-3 digit) if they're not already in the reference
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
            # Update valid_codes with all hierarchical codes
            valid_codes = all_codes
            
            # Update code_to_title for parent codes that don't have titles
            for code in all_codes:
                if code not in code_to_title:
                    if len(code) == 1:
                        code_to_title[code] = f"Major Group {code}"
                    elif len(code) == 2:
                        code_to_title[code] = f"Sub-Major Group {code}"
                    elif len(code) == 3:
                        code_to_title[code] = f"Minor Group {code}"
        
        logger.info(f"Loaded {len(valid_codes)} valid ISCO-08 codes from reference file")
        return isco_df, valid_codes, code_to_title
        
    except Exception as e:
        logger.warning(f"Error loading ISCO-08 reference file: {e}")
        return None, valid_codes, code_to_title

def load_and_map_labels(processed_dir, config=None):
    """
    Load training and validation data and create label mapping
    that includes ALL valid ISCO codes, even those not in training
    
    Args:
        processed_dir (str): Directory with processed data
        config (dict, optional): Configuration dictionary
        
    Returns:
        tuple: Train DataFrame, validation DataFrame, label to ID mapping, code metadata
    """
    # Load train and validation data
    train_path = os.path.join(processed_dir, "train.csv")
    val_path = os.path.join(processed_dir, "val.csv")
    
    # Attempt to load test data as well if it exists
    test_path = os.path.join(processed_dir, "test.csv")
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Load ISCO reference data with config
    isco_ref_df, valid_reference_codes, code_to_title = load_isco_reference(config)
    
    # Check if we successfully loaded reference data
    if valid_reference_codes:
        logger.info(f"Using {len(valid_reference_codes)} valid ISCO-08 codes from official reference")
        # Convert set to list and sort
        all_valid_codes = sorted(list(valid_reference_codes))
    else:
        # If reference file can't be loaded, use the fallback function
        logger.warning("ISCO reference data not available, using fallback code generation")
        all_valid_codes = generate_fallback_isco_codes()
    
    # Also explicitly add all observed codes from all data sources
    all_observed_codes = set()
    
    # Add codes from training data
    all_observed_codes.update(train_df["isco_code"].astype(str).unique())
    
    # Add codes from validation data
    all_observed_codes.update(val_df["isco_code"].astype(str).unique())
    
    # Add test data codes if available
    if os.path.exists(test_path):
        try:
            test_df = pd.read_csv(test_path)
            all_observed_codes.update(test_df["isco_code"].astype(str).unique())
            logger.info(f"Added {len(test_df['isco_code'].unique())} unique ISCO codes from test data")
        except Exception as e:
            logger.warning(f"Could not load test set for label mapping: {e}")
    
    # Log the number of observed codes 
    logger.info(f"Found {len(all_observed_codes)} unique ISCO codes in processed data")
    
    # We've already validated against the reference during preprocessing
    # Any code still in the processed data is valid, so we won't filter or 
    # issue additional warnings here
    logger.info(f"Using {len(all_valid_codes)} ISCO codes from reference and {len(all_observed_codes)} observed codes")
    
    # Combine all valid codes from reference with observed codes in processed data
    all_codes = set(all_valid_codes)
    
    # Add any validated observed codes that weren't in the reference
    for code in all_observed_codes:
        code_str = str(code).strip()
        
        # Skip empty codes
        if not code_str:
            continue
            
        # Skip known invalid codes
        if code_str in ['9999', '999999']:
            continue
            
        # Skip non-numeric codes
        if not code_str.isdigit():
            continue
            
        # Skip codes longer than 4 digits
        if len(code_str) > 4:
            continue
            
        # Add valid code
        all_codes.add(code_str)
    
    # First normalize all codes to handle armed forces codes and other special cases
    normalized_codes = set()
    armed_forces_mappings = {}  # Track mappings for armed forces codes
    
    for code in all_codes:
        code_str = str(code)
        # Handle armed forces codes without leading zeros
        if len(code_str) == 3 and code_str in ['110', '210', '310']:
            normalized_code = '0' + code_str  # Add leading zero for armed forces
            normalized_codes.add(normalized_code)
            armed_forces_mappings[code_str] = normalized_code  # Store mapping
            logger.debug(f"Normalized armed forces code {code_str} to {normalized_code}")
        else:
            normalized_codes.add(code_str)
    
    # Replace all_codes with normalized versions
    all_codes = normalized_codes
    
    # Also normalize the observed codes to maintain consistency
    normalized_observed = set()
    for code in all_observed_codes:
        code_str = str(code).strip()
        if code_str in armed_forces_mappings:
            # Replace with normalized version
            normalized_observed.add(armed_forces_mappings[code_str])
        else:
            normalized_observed.add(code_str)
    
    # Replace observed codes with normalized versions
    all_observed_codes = normalized_observed
    
    # Check if we should filter to only 4-digit codes (from config)
    four_digit_only = config and config.get("model", {}).get("four_digit_only", False)
    
    # Create final filtered set
    if four_digit_only:
        # Keep only 4-digit codes for classification and include armed forces special codes
        all_filtered_codes = {code for code in all_codes if len(code) == 4 or 
                            (code.startswith('0') and code in ['0110', '0210', '0310'])}
        logger.info(f"Filtered to {len(all_filtered_codes)} 4-digit ISCO codes only (including armed forces special codes, removed {len(all_codes) - len(all_filtered_codes)} non-4-digit codes)")
    else:
        # Keep all codes (including hierarchy levels)
        all_filtered_codes = all_codes
        logger.info(f"Using all {len(all_filtered_codes)} ISCO codes (including hierarchical levels 1-4 digits)")
        
    # Always verify we have codes for the observed data
    # Normalize any armed forces codes in the observed data
    normalized_missing = []
    for code in all_observed_codes:
        code_str = str(code).strip()
        # If the original code doesn't match, check if a normalized version does
        if code_str and code_str not in all_filtered_codes:
            # Check if this is an armed forces code
            if len(code_str) == 3 and code_str in ['110', '210', '310']:
                normalized = '0' + code_str
                # If the normalized version is missing too, then it's truly missing
                if normalized not in all_filtered_codes:
                    normalized_missing.append(code_str)
            else:
                normalized_missing.append(code_str)
                
    missing_observed = normalized_missing
    
    if missing_observed:
        logger.warning(f"Warning: {len(missing_observed)} observed codes are missing from our filtered set")
        if four_digit_only:
            non_four_digit = [code for code in missing_observed if len(str(code)) != 4 and not (len(str(code)) == 3 and str(code) in ['110', '210', '310'])]
            if non_four_digit:
                logger.info(f"This is expected as we're filtering to 4-digit codes only (found {len(non_four_digit)} non-4-digit codes in data)")
                
        # Log sample of missing codes
        sample = missing_observed[:min(10, len(missing_observed))]
        logger.info(f"Sample missing codes: {', '.join(str(c) for c in sample)}")
    
    # Create sorted list of all codes
    unique_labels = sorted(list(all_filtered_codes))
    
    # Create mapping
    label2id = {str(label): idx for idx, label in enumerate(unique_labels)}
    
    # Create metadata for codes
    code_metadata = {}
    for code in unique_labels:
        # Initialize metadata
        metadata = {
            "title": code_to_title.get(code, f"Code {code}"),
            "level": len(code)  # 1-digit to 4-digit
        }
        
        # Add hierarchy information if possible
        if len(code) > 1:
            metadata["parent_code"] = code[0] if len(code) == 2 else code[:-1]
        
        # Store metadata
        code_metadata[code] = metadata
    
    # Verify all observed codes are in the mapping
    missing_codes = set()
    for code in all_observed_codes:
        code_str = str(code).strip()
        if code_str and code_str not in ['9999', '999999'] and code_str.isdigit() and len(code_str) <= 4:
            # Check if the code needs normalization (armed forces codes)
            normalized_code = code_str
            if len(code_str) == 3 and code_str in ['110', '210', '310']:
                normalized_code = '0' + code_str
            
            # Now check if either the original or normalized code is in the mapping
            if normalized_code not in label2id and code_str not in label2id:
                missing_codes.add(code_str)
    
    if missing_codes:
        logger.warning(f"Missing {len(missing_codes)} valid observed codes in mapping: {missing_codes}")
    
    logger.info(f"Created comprehensive label mapping with {len(label2id)} ISCO codes")
    return train_df, val_df, label2id, code_metadata

def prepare_datasets(train_df, val_df, label2id, max_seq_length):
    """
    Prepare train and validation datasets
    
    Args:
        train_df (DataFrame): Training data
        val_df (DataFrame): Validation data
        label2id (dict): Label to ID mapping
        max_seq_length (int): Maximum sequence length
        
    Returns:
        tuple: Train dataset, validation dataset
    """
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    # Check for 4-digit-only mode by looking at all keys that are ISCO codes
    isco_keys = [k for k in label2id.keys() if str(k).isdigit()]
    valid_length_keys = [k for k in isco_keys if len(str(k)) == 4 or (len(str(k)) < 4 and str(k).startswith('0'))]
    four_digit_only = len(valid_length_keys) == len(isco_keys)
    
    if four_digit_only:
        logger.info("Using 4-digit-only mode for dataset preparation")
    
    # Helper function to normalize ISCO codes to 4 digits
    def normalize_isco_code(code):
        code_str = str(code)
        # If it's already 4 digits, return as is
        if len(code_str) == 4:
            return code_str
        # Special handling for armed forces codes (these are priority)
        if len(code_str) == 3 and code_str in ['110', '210', '310']:
            armed_forces = '0' + code_str  # Convert to proper armed forces code format
            # Ensure this exists in our labels
            if armed_forces in label2id:
                return armed_forces
            # If not, check if the original code is in the mapping
            if code_str in label2id:
                return code_str
        # If it's shorter and potentially a code with leading zeros
        elif len(code_str) < 4:
            # First check if zero-padded version exists in mapping
            padded = code_str.zfill(4)
            if padded in label2id:
                return padded
        # Return original if no normalization needed/possible
        return code_str
    
    # Filter and convert labels to IDs, handling codes not in mapping
    train_texts = []
    train_labels = []
    skipped_formats = set()  # Track skipped format types
    
    for i, label in enumerate(train_df["isco_code"].tolist()):
        # Normalize the code to handle potential leading zeros
        label_str = normalize_isco_code(label)
        
        # Skip non-4-digit codes if in 4-digit-only mode
        if four_digit_only and len(label_str) != 4:
            skipped_formats.add(len(label_str))
            continue
            
        # Only include this sample if its label is in our mapping
        if label_str in label2id:
            train_texts.append(train_df["text"].iloc[i])
            train_labels.append(label2id[label_str])
        else:
            # Try zero-padding shorter codes to see if they match
            if len(label_str) < 4:
                padded = label_str.zfill(4)
                if padded in label2id:
                    train_texts.append(train_df["text"].iloc[i])
                    train_labels.append(label2id[padded])
    
    # Do the same for validation data
    val_texts = []
    val_labels = []
    for i, label in enumerate(val_df["isco_code"].tolist()):
        # Normalize the code to handle potential leading zeros
        label_str = normalize_isco_code(label)
        
        # Skip non-4-digit codes if in 4-digit-only mode
        if four_digit_only and len(label_str) != 4:
            continue
            
        # Only include this sample if its label is in our mapping
        if label_str in label2id:
            val_texts.append(val_df["text"].iloc[i])
            val_labels.append(label2id[label_str])
        else:
            # Try zero-padding shorter codes to see if they match
            if len(label_str) < 4:
                padded = label_str.zfill(4)
                if padded in label2id:
                    val_texts.append(val_df["text"].iloc[i])
                    val_labels.append(label2id[padded])
    
    # Log info about skipped formats if any                
    if skipped_formats and four_digit_only:
        logger.info(f"Skipped non-4-digit codes with these lengths: {sorted(skipped_formats)}")
    
    # Log dataset stats
    logger.info(f"Training with {len(label2id)} unique ISCO codes")
    logger.info(f"Prepared {len(train_texts)} training samples and {len(val_texts)} validation samples")
    
    # Create datasets
    train_dataset = ISCODataset(train_texts, train_labels, tokenizer, max_seq_length)
    val_dataset = ISCODataset(val_texts, val_labels, tokenizer, max_seq_length)
    
    # Verify we have data
    if len(train_dataset) == 0:
        raise ValueError("No training samples left after filtering. Check if your label mapping matches your data.")
    if len(val_dataset) == 0:
        raise ValueError("No validation samples left after filtering. Check if your label mapping matches your data.")
    
    return train_dataset, val_dataset

def initialize_model(model_name, num_labels):
    """
    Initialize RoBERTa model for classification
    
    Args:
        model_name (str): Name of the pretrained model
        num_labels (int): Number of labels for classification
        
    Returns:
        RobertaForSequenceClassification: Initialized model
    """
    device = get_device()
    
    # Initialize model with the number of labels
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    ).to(device)
    
    return model

def train_model_with_params(model, train_dataset, val_dataset, config):
    """
    Train model with parameters
    
    Args:
        model: The model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config (dict): Configuration dictionary
        
    Returns:
        Trainer: Trained model trainer
    """
    from colorama import Fore, Style
    # Create output path for model runs
    run_dir = os.path.join(config["output"]["model_dir"], "runs")
    ensure_dir(run_dir)
    
    # Get device with large dataset awareness
    dataset_size = len(train_dataset)
    logger.info(f"Training dataset size: {dataset_size} examples")
    
    # For extremely large datasets, consider falling back to CPU to avoid OOM errors
    force_cpu = False
    if dataset_size > 150000:
        logger.warning("Very large dataset detected. Checking if we should use CPU instead of GPU...")
        try:
            # Check available system memory
            import psutil
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024 ** 3)
            
            # If we have enough system memory, consider CPU training for stability
            if available_gb > 12:  # Need plenty of RAM for CPU training
                logger.warning(f"Available memory: {available_gb:.1f} GB. Consider CPU training for very large datasets.")
                user_prompt = "You have a very large dataset. Would you like to use CPU instead of GPU for more stable training? (y/n): "
                try:
                    use_cpu = input(user_prompt).strip().lower() == 'y'
                    if use_cpu:
                        force_cpu = True
                        logger.warning("Using CPU for training as requested.")
                    else:
                        logger.info("Continuing with GPU (MPS/CUDA) as requested.")
                except:
                    # In non-interactive mode, make a decision based on dataset size
                    if dataset_size > 200000:
                        force_cpu = True
                        logger.warning("Non-interactive mode: Using CPU for extremely large dataset.")
            else:
                logger.warning(f"Low available memory ({available_gb:.1f} GB). Will try GPU training but may encounter OOM errors.")
        except ImportError:
            logger.warning("Could not check system memory (psutil not installed). Continuing with GPU.")
    
    # Get appropriate device based on settings
    if force_cpu:
        device = torch.device("cpu")
        logger.info("Using CPU device for training")
        os.environ["FORCE_CPU"] = "1"  # Force CPU for all operations
    else:
        device = get_device()
        logger.info(f"Using {device} device for training")
    
    # Configure MPS memory settings if enabled
    if device.type == "mps" and config["training"].get("mps_memory_efficient", True):
        configure_mps_memory(mps_memory_efficient=True)
    
    # Configure training arguments
    # Note: fp16 is not supported on MPS, only enable it on CUDA
    fp16_enabled = config["training"]["enable_optimizations"] and device.type == "cuda"
    
    # For early stopping to work, we need to enable model saving and loading
    # Get early stopping setting from config
    early_stopping_enabled = config["training"].get("early_stopping_patience", 0) > 0
    
    # Use the configured values for previously hardcoded parameters
    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=config["model"]["epochs"],
        per_device_train_batch_size=config["model"]["batch_size"],
        per_device_eval_batch_size=config["model"]["batch_size"],
        warmup_steps=config["training"].get("warmup_steps", 500),
        weight_decay=config["training"].get("weight_decay", 0.01),
        logging_dir=os.path.join(run_dir, "logs"),
        logging_steps=config["training"].get("logging_steps", 10),
        evaluation_strategy=config["training"].get("evaluation_strategy", "epoch"),
        # If early stopping is enabled, we need to save checkpoints
        save_strategy="epoch" if early_stopping_enabled else config["training"].get("save_strategy", "no"),
        save_total_limit=config["training"].get("save_total_limit", 1),  # Only keep one checkpoint
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
        # Enable load_best_model_at_end if early stopping is enabled
        load_best_model_at_end=early_stopping_enabled,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=config["model"]["learning_rate"],
        save_only_model=True,  # Save only model weights, not optimizer state
        save_on_each_node=False,  # Don't save separately on each node
        fp16=fp16_enabled,  # Only enable on CUDA devices
        gradient_checkpointing=config["training"]["enable_optimizations"],
        dataloader_num_workers=config["training"].get("dataloader_num_workers", 2),
    )
    
    # Store the training config in args for our custom trainer to access
    training_args.training_config = config["training"]
    
    # Set up callbacks - only include early stopping if enabled
    callbacks = []
    
    # Add early stopping callback if enabled
    if early_stopping_enabled:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config["training"].get("early_stopping_patience", 2)
            )
        )
        logger.info(f"Early stopping enabled with patience {config['training'].get('early_stopping_patience', 2)}")
    else:
        logger.info("Early stopping disabled")
        
    # Choose appropriate trainer based on device
    if device.type == "mps" and config["training"].get("mps_mixed_precision", True):
        logger.info("Using MPS-optimized trainer with mixed precision")
        trainer = MPSOptimizedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks
        )
    else:
        # Use standard trainer for CUDA or when MPS optimizations are disabled
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks
        )
    
    # Display training summary
    dataset_size = len(train_dataset)
    num_epochs = config["model"]["epochs"]
    batch_size = config["model"]["batch_size"]
    steps_per_epoch = dataset_size // (batch_size * config["training"].get("gradient_accumulation_steps", 1))
    total_steps = steps_per_epoch * num_epochs
    
    logger.info(f"{Fore.CYAN}Training Configuration:{Style.RESET_ALL}")
    logger.info(f"  • Dataset size: {dataset_size:,} examples")
    logger.info(f"  • Batch size: {batch_size}")
    logger.info(f"  • Gradient accumulation steps: {config['training'].get('gradient_accumulation_steps', 1)}")
    logger.info(f"  • Epochs: {num_epochs}")
    logger.info(f"  • Learning rate: {config['model']['learning_rate']}")
    logger.info(f"  • Total training steps: ~{total_steps:,}")
    logger.info(f"  • Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train the model
    logger.info(f"{Fore.CYAN}{Style.BRIGHT}Starting model training{Style.RESET_ALL}")
    
    import time
    start_time = time.time()
    
    trainer.train()
    
    # Calculate training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"{Fore.GREEN}{Style.BRIGHT}Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s{Style.RESET_ALL}")
    
    return trainer

def save_best_model(trainer, best_model_dir, label2id, code_metadata=None):
    """
    Save the best model and label mappings
    
    Args:
        trainer: Trained model trainer
        best_model_dir (str): Directory to save the best model
        label2id (dict): Label to ID mapping
        code_metadata (dict, optional): Metadata for ISCO codes
    """
    # Ensure directory exists
    ensure_dir(best_model_dir)
    
    # Create id2label mapping
    id2label = {idx: label for label, idx in label2id.items()}
    
    # Log mapping statistics
    logger.info(f"Model can predict {len(label2id)} unique ISCO codes")
    
    # Update model config with label mappings to ensure they are in config.json
    # This makes config.json the single source of truth for mappings
    model = trainer.model
    model.config.id2label = {str(idx): str(label) for idx, label in id2label.items()}
    model.config.label2id = {str(label): idx for label, idx in label2id.items()}
    
    # Save model with updated config
    logger.info(f"Saving best model to {best_model_dir}")
    trainer.save_model(best_model_dir)
    
    # Save tokenizer explicitly
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    tokenizer.save_pretrained(best_model_dir)
    
    # Also save label mappings as separate files for backward compatibility
    with open(os.path.join(best_model_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f)
    
    with open(os.path.join(best_model_dir, "id2label.json"), "w") as f:
        json.dump(id2label, f)
        
    # Save a complete list of all ISCO codes in the model
    with open(os.path.join(best_model_dir, "all_isco_codes.json"), "w") as f:
        all_codes = sorted(list(label2id.keys()))
        json.dump(all_codes, f, indent=2)
    
    # Save code metadata if available
    if code_metadata:
        with open(os.path.join(best_model_dir, "code_metadata.json"), "w") as f:
            json.dump(code_metadata, f, indent=2)
            
        # Create a hierarchical structure of codes
        hierarchy = {}
        for code, metadata in code_metadata.items():
            level = metadata.get("level", len(code))
            
            # Skip invalid levels
            if level < 1 or level > 4:
                continue
                
            # Initialize level in hierarchy if needed
            if level not in hierarchy:
                hierarchy[level] = {}
                
            # Major group (1-digit)
            if level == 1:
                hierarchy[level][code] = {
                    "title": metadata.get("title", f"Code {code}"),
                    "children": {}
                }
            # Sub-major group (2-digit)
            elif level == 2:
                parent = code[0]
                if parent in hierarchy.get(1, {}):
                    if "children" not in hierarchy[1][parent]:
                        hierarchy[1][parent]["children"] = {}
                    hierarchy[1][parent]["children"][code] = {
                        "title": metadata.get("title", f"Code {code}"),
                        "children": {}
                    }
            # Minor group (3-digit)
            elif level == 3:
                parent = code[:2]
                major = code[0]
                if major in hierarchy.get(1, {}) and parent in hierarchy.get(1, {}).get(major, {}).get("children", {}):
                    if "children" not in hierarchy[1][major]["children"][parent]:
                        hierarchy[1][major]["children"][parent]["children"] = {}
                    hierarchy[1][major]["children"][parent]["children"][code] = {
                        "title": metadata.get("title", f"Code {code}"),
                        "children": {}
                    }
            # Unit group (4-digit)
            elif level == 4:
                minor = code[:3]
                submajor = code[:2]
                major = code[0]
                if major in hierarchy.get(1, {}) and submajor in hierarchy.get(1, {}).get(major, {}).get("children", {}):
                    parent_minor = hierarchy[1][major]["children"][submajor]["children"]
                    if minor in parent_minor:
                        if "children" not in parent_minor[minor]:
                            parent_minor[minor]["children"] = {}
                        parent_minor[minor]["children"][code] = {
                            "title": metadata.get("title", f"Code {code}")
                        }
        
        # Save hierarchy
        with open(os.path.join(best_model_dir, "code_hierarchy.json"), "w") as f:
            json.dump(hierarchy, f, indent=2)

def train_model(config):
    """
    Train ISCO classification model
    
    Args:
        config (dict): Configuration dictionary
    """
    # Load data and create label mapping
    train_df, val_df, label2id, code_metadata = load_and_map_labels(config["data"]["processed_dir"], config)
    
    # Log mapping statistics
    logger.info(f"Model will be trained with {len(label2id)} unique ISCO codes")
    logger.info(f"Training data contains {train_df['isco_code'].nunique()} unique ISCO codes")
    logger.info(f"Validation data contains {val_df['isco_code'].nunique()} unique ISCO codes")
    logger.info(f"Training data size: {len(train_df)} examples")
    
    # Dataset size warning and option to train on subset
    dataset_size = len(train_df)
    if dataset_size > 50000:
        from colorama import Fore, Style
        
        # Format a user-friendly warning message
        warning_message = f"\n{Fore.YELLOW}{Style.BRIGHT}⚠️ Large Dataset Warning ⚠️{Style.RESET_ALL}\n"
        warning_message += f"{Fore.YELLOW}You are about to train on a large dataset with {dataset_size:,} records.{Style.RESET_ALL}\n"
        warning_message += f"{Fore.YELLOW}This may require significant memory and could take several hours.{Style.RESET_ALL}\n\n"
        warning_message += "Options:\n"
        warning_message += "1) Proceed with full dataset (recommended for production models)\n"
        warning_message += "2) Train on a random subset (faster, uses less memory, but may reduce accuracy)\n"
        
        print(warning_message)
        
        # Prompt user for choice
        try:
            choice = input("Enter your choice (1 or 2): ").strip()
            
            if choice == "2":
                # Calculate a reasonable subset size based on original size
                if dataset_size > 150000:
                    subset_size = 50000
                elif dataset_size > 100000:
                    subset_size = 40000
                else:
                    subset_size = 30000
                
                # Get user input for custom subset size
                try:
                    custom_size = input(f"Enter subset size (default: {subset_size:,}): ").strip()
                    if custom_size and custom_size.isdigit():
                        subset_size = min(int(custom_size), dataset_size)
                except:
                    pass  # Use default if there's any error
                
                # Sample the dataset
                logger.info(f"{Fore.CYAN}Using a random subset of {subset_size:,} records (from {dataset_size:,}){Style.RESET_ALL}")
                train_df = train_df.sample(subset_size, random_state=42)
                logger.info(f"New training data size: {len(train_df)} examples")
            else:
                logger.info(f"{Fore.GREEN}Proceeding with full dataset ({dataset_size:,} records){Style.RESET_ALL}")
        except:
            # In non-interactive mode, use the full dataset
            logger.info("Non-interactive mode: proceeding with full dataset")
    
    # Automatically adjust training parameters for very large datasets
    original_batch_size = config["model"]["batch_size"]
    original_grad_accum = config["training"]["gradient_accumulation_steps"]
    
    # For large datasets, decrease batch size and increase gradient accumulation
    if len(train_df) > 100000:
        # Only adjust if not already adjusted (batch size not too small)
        if config["model"]["batch_size"] > 8:
            logger.warning("Very large dataset detected. Automatically adjusting training parameters.")
            # Reduce batch size
            config["model"]["batch_size"] = max(8, config["model"]["batch_size"] // 2)
            # Increase gradient accumulation to compensate
            config["training"]["gradient_accumulation_steps"] = config["training"]["gradient_accumulation_steps"] * 2
            
            logger.info(f"Adjusted batch size from {original_batch_size} to {config['model']['batch_size']}")
            logger.info(f"Adjusted gradient accumulation steps from {original_grad_accum} to {config['training']['gradient_accumulation_steps']}")
    
    # Check for codes in validation but not in training
    val_only_codes = set(val_df["isco_code"].astype(str).unique()) - set(train_df["isco_code"].astype(str).unique())
    if val_only_codes:
        logger.warning(f"Found {len(val_only_codes)} ISCO codes in validation data but not in training data")
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(
        train_df, val_df, label2id, config["model"]["max_seq_length"]
    )
    
    # Initialize model
    model = initialize_model(config["model"]["name"], len(label2id))
    
    # Train model
    trainer = train_model_with_params(model, train_dataset, val_dataset, config)
    
    # Since we disabled checkpointing, explicitly save the final model
    logger.info("Training complete. Saving final model...")
    
    # Create a timestamped directory for this specific run's checkpoint
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_checkpoint_dir = os.path.join(config["output"]["model_dir"], f"checkpoint-final-{timestamp}")
    ensure_dir(final_checkpoint_dir)
    
    # Save the model to the timestamped directory
    trainer.save_model(final_checkpoint_dir)
    logger.info(f"Final model saved to {final_checkpoint_dir}")
    
    # Save to best_model directory with code metadata
    save_best_model(trainer, config["output"]["best_model_dir"], label2id, code_metadata)
    
    return trainer, label2id, code_metadata

if __name__ == "__main__":
    # Load config
    config = load_config()
    
    # Train model
    trainer, label2id, code_metadata = train_model(config)
    
    # Log number of codes at each level
    level_counts = {}
    for code, metadata in code_metadata.items():
        level = metadata.get("level", len(code))
        if level not in level_counts:
            level_counts[level] = 0
        level_counts[level] += 1
    
    # Print statistics about the model
    logger.info("=== ISCO Code Classification Model Statistics ===")
    logger.info(f"Total number of ISCO codes: {len(label2id)}")
    
    # Print codes by level
    for level, count in sorted(level_counts.items()):
        level_name = {
            1: "Major Groups (1-digit)",
            2: "Sub-Major Groups (2-digit)",
            3: "Minor Groups (3-digit)",
            4: "Unit Groups (4-digit)"
        }.get(level, f"Level-{level} codes")
        
        logger.info(f"  - {level_name}: {count} codes")
    
    logger.info("Model training completed successfully!")