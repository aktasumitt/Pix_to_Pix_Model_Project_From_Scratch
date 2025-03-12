class Config:
    # Data Ingestion Stage
    local_real_data_path = "local_data/real_images"
    save_real_data_path = "artifacts/data_ingestion/real_data_paths.json"
    local_edge_data_path = "local_data/edge_images/avg"
    save_edge_data_path = "artifacts/data_ingestion/edge_data_paths.json"

    # Data Transformation Stage
    train_dataset_save_path = "artifacts/data_transformation/train_dataset.pth"
    valid_dataset_save_path = "artifacts/data_transformation/valid_dataset.pth"
    test_dataset_save_path = "artifacts/data_transformation/test_dataset.pth"

    # Model Ingestion Stage
    generator_save_path = "artifacts/model/generator.pth"
    discriminator_save_path = "artifacts/model/discriminator.pth"

    # Training Stage
    checkpoint_save_path = "callbacks/checkpoints/checkpoint_latest.pth.tar"
    final_discriminator_model_path = "callbacks/final_model/discriminator.pth"
    final_generator_model_path = "callbacks/final_model/generator.pth"
    results_save_path = "results/train_results.json"

    # Testing Stage
    test_generated_img_path = "results/test/test_predict_images.jpg"
    test_result_save_path = "results/test/test_results.json"
    test_model_generator_save_path = "callbacks/test_models/tested_generator.pth"
    test_model_discriminator_save_path = "callbacks/test_models/tested_discriminator.pth"

    # Prediction Stage
    predicted_img_save_path = "prediction_images/generated_images"
    prediction_img_load_path = "prediction_images/edge_images"
