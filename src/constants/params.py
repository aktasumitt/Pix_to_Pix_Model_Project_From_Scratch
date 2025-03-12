class Params:

    # Data Transformation Stage
    valid_rate = 0.2
    test_rate = 0.1
    resize_size = 286
    crop_size = 256

    # Model Ingestion Stage
    channel_size_in = 1
    channel_size_out = 3
    
    
    # Training Stage
    batch_size = 50
    device = "cuda"
    learning_rate = 5e-4
    betas = (0.5,0.999)
    epochs = 20
    load_checkpoint = False

    # Test Step
    load_checkpoints_for_test = False
    save_tested_model = False

    # Prediction step
    img_size = 256
