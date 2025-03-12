from src.entity.config_entity import (TestConfig,
                                      TrainingConfig,
                                      DataIngestionConfig,
                                      ModelIngestionConfig,
                                      DataTransformationConfig,
                                      PredictionConfig)

from src.constants.config import Config
from src.constants.params import Params

class Configuration():

    def __init__(self):
        self.config = Config
        self.params = Params

    def data_ingestion_config(self):
        return DataIngestionConfig(
            local_real_data_path=self.config.local_real_data_path,
            local_edge_data_path=self.config.local_edge_data_path,
            save_edge_data_path=self.config.save_edge_data_path,
            save_real_data_path=self.config.save_real_data_path
        )

    def data_transformation_config(self):
        return DataTransformationConfig(
            train_dataset_save_path=self.config.train_dataset_save_path,
            valid_dataset_save_path=self.config.valid_dataset_save_path,
            test_dataset_save_path=self.config.test_dataset_save_path,
            test_rate=self.params.test_rate,
            valid_rate=self.params.valid_rate,
            real_img_path_list=self.config.save_real_data_path,
            edge_img_path_list=self.config.save_edge_data_path,
            resize_size=self.params.resize_size,
            crop_size=self.params.crop_size
        )

    def model_config(self):
        return ModelIngestionConfig(
            generator_save_path=self.config.generator_save_path,
            discriminator_save_path=self.config.discriminator_save_path,
            channel_size_in=self.params.channel_size_in,
            channel_size_out=self.params.channel_size_out
        )

    def training_config(self):
        return TrainingConfig(
            generator_model_path=self.config.generator_save_path,
            discriminator_model_path=self.config.discriminator_save_path,
            train_dataset_path=self.config.train_dataset_save_path,
            valid_dataset_path=self.config.valid_dataset_save_path,
            checkpoint_path=self.config.checkpoint_save_path,
            final_generator_model_path=self.config.final_generator_model_path,
            final_discriminator_model_path=self.config.final_discriminator_model_path,
            results_save_path=self.config.results_save_path,
            batch_size=self.params.batch_size,
            device=self.params.device,
            learning_rate=self.params.learning_rate,
            betas=self.params.betas,
            epochs=self.params.epochs,
            load_checkpoint=self.params.load_checkpoint
        )

    def test_config(self):
        return TestConfig(
            final_model_generator_path=self.config.final_generator_model_path,
            final_model_discriminator_path=self.config.final_discriminator_model_path,
            test_dataset_path=self.config.test_dataset_save_path,
            batch_size=self.params.batch_size,
            load_checkpoints_for_test=self.params.load_checkpoints_for_test,
            checkpoint_path=self.config.checkpoint_save_path,
            test_result_save_path=self.config.test_result_save_path,
            test_generated_image_path=self.config.test_generated_img_path,
            test_model_generator_save_path=self.config.test_model_generator_save_path,
            test_model_discriminator_save_path=self.config.test_model_discriminator_save_path,
            save_tested_model=self.params.save_tested_model
        )

    def prediction_config(self):
        return PredictionConfig(
            img_size=self.params.img_size,
            generator_path=self.config.final_generator_model_path,
            predicted_img_save_path=self.config.predicted_img_save_path,
            prediction_img_load_path=self.config.prediction_img_load_path
        )