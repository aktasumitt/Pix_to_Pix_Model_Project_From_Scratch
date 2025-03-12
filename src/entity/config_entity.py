from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    local_real_data_path: Path
    save_real_data_path: Path
    local_edge_data_path: Path
    save_edge_data_path: Path


@dataclass
class DataTransformationConfig:
    real_img_path_list: Path
    edge_img_path_list: Path
    train_dataset_save_path: Path
    valid_dataset_save_path: Path
    test_dataset_save_path: Path
    test_rate: float
    valid_rate: float
    resize_size: int
    crop_size: int


@dataclass
class ModelIngestionConfig:
    generator_save_path: Path
    discriminator_save_path: Path
    channel_size_in: int
    channel_size_out: int


@dataclass
class TrainingConfig:
    generator_model_path: Path
    discriminator_model_path: Path
    train_dataset_path: Path
    valid_dataset_path: Path
    checkpoint_path: Path
    final_generator_model_path: Path
    final_discriminator_model_path: Path
    results_save_path: Path
    batch_size: int
    device: str
    learning_rate: float
    betas: float
    epochs: int
    load_checkpoint: bool


@dataclass
class TestConfig:
    final_model_generator_path: Path
    final_model_discriminator_path: Path
    test_dataset_path: Path
    batch_size: int
    load_checkpoints_for_test: bool
    checkpoint_path: Path
    test_result_save_path: Path
    test_generated_image_path: Path
    test_model_generator_save_path: Path
    test_model_discriminator_save_path: Path
    save_tested_model: bool


@dataclass
class PredictionConfig:
    img_size: int
    generator_path: Path
    predicted_img_save_path: Path
    prediction_img_load_path: Path
