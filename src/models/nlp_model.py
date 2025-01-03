from google.cloud import aiplatform
import pandas as pd


def train_nlp_model(training_data_path, model_display_name, project_id, location, staging_bucket):
    """Trains an NLP model (e.g., sentiment analysis, topic modeling) using Vertex AI.

    Args:
        training_data_path: Path to the training data (e.g., text data and labels in CSV format).
        model_display_name: Display name for the trained model.
        project_id: Google Cloud project ID.
        location: Region where Vertex AI resources are deployed.
        staging_bucket: GCS bucket for storing model artifacts.
    """
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=location, staging_bucket=staging_bucket)

    # Upload training data to GCS
    dataset = aiplatform.TabularDataset.create(
        display_name=f"{model_display_name}_dataset",
        gcs_source=[training_data_path],  # Replace with the GCS path to your training data
    )

    # Create a custom training job for the NLP model
    training_job = aiplatform.CustomTrainingJob(
        display_name=f"{model_display_name}_training_job",
        script_path="path/to/your/training_script.py",  # Replace with the actual path to your training script
        container_uri="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-8:latest",  # Use a suitable container for your framework
        model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest",  # Serving container for prediction
    )

    # Run the training job
    model = training_job.run(
        dataset=dataset,
        model_display_name=model_display_name,
        args=["--epochs", "10", "--batch_size", "32"],  # Additional arguments for the training script
        replica_count=1,
        machine_type="n1-standard-4",  # Adjust based on resource requirements
    )

    print(f"Model training completed. Model uploaded as {model.resource_name}.")


def predict_sentiment(model_endpoint, text, project_id, location):
    """Predicts sentiment of the given text.

    Args:
        model_endpoint: Endpoint of the deployed sentiment analysis model.
        text: Text to analyze.
        project_id: Google Cloud project ID.
        location: Region where Vertex AI resources are deployed.

    Returns:
        Predicted sentiment (e.g., positive, negative, neutral).
    """
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=location)

    # Create a prediction client
    endpoint = aiplatform.Endpoint(endpoint_name=model_endpoint)

    # Make predictions
    predictions = endpoint.predict(
        instances=[
            {"text": text}  # Replace with the correct input format for your model
        ]
    )

    # Extract predicted sentiment from the predictions
    predicted_sentiment = predictions.predictions[0]["sentiment"]  # Adjust based on your model's response schema
    print(f"Predicted sentiment: {predicted_sentiment}")
    return predicted_sentiment


if __name__ == "__main__":
    # Example usage

    # Training parameters
    training_data_path = "gs://your-bucket/path/to/training_data.csv"  # Replace with your GCS training data path
    model_display_name = "nlp_sentiment_model"
    project_id = "your-gcp-project-id"  # Replace with your GCP project ID
    location = "us-central1"  # Replace with your preferred region
    staging_bucket = "gs://your-staging-bucket"  # Replace with your GCS staging bucket

    # Train the model
    train_nlp_model(
        training_data_path=training_data_path,
        model_display_name=model_display_name,
        project_id=project_id,
        location=location,
        staging_bucket=staging_bucket,
    )

    # Prediction parameters
    model_endpoint = "projects/your-project/locations/us-central1/endpoints/your-endpoint-id"  # Replace with your model's endpoint
    sample_text = "I loved the movie! It was amazing."  # Replace with the text to analyze

    # Generate sentiment predictions
    sentiment = predict_sentiment(
        model_endpoint=model_endpoint,
        text=sample_text,
        project_id=project_id,
        location=location,
    )
    print(f"Sentiment for the text '{sample_text}': {sentiment}")

