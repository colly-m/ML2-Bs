from google.cloud import aiplatform
import pandas as pd


def train_recommendation_model(training_data_path, model_display_name, project_id, location, staging_bucket):
    """Trains a recommendation model using Vertex AI.

    Args:
        training_data_path: Path to the training data (e.g., CSV file with user-item interactions).
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
        gcs_source=[training_data_path],
    )

    # Create a custom training job for the recommendation model
    training_job = aiplatform.CustomTrainingJob(
        display_name=f"{model_display_name}_training_job",
        script_path="path/to/your/training_script.py",  # Your model training script
        container_uri="us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.0-24:latest",
        model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest",
    )

    # Run the training job
    model = training_job.run(
        dataset=dataset,
        model_display_name=model_display_name,
        args=["--epochs", "50", "--batch_size", "64"],  # Pass additional arguments to your training script
        replica_count=1,
        machine_type="n1-standard-4",
    )

    print(f"Model training completed. Model uploaded as {model.resource_name}.")


def predict_recommendations(model_endpoint, user_id, project_id, location):
    """Generates recommendations for a given user.

    Args:
        model_endpoint: Endpoint of the deployed recommendation model.
        user_id: ID of the user for whom to generate recommendations.
        project_id: Google Cloud project ID.
        location: Region where Vertex AI resources are deployed.

    Returns:
        List of recommended items.
    """
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=location)

    # Create a prediction client
    endpoint = aiplatform.Endpoint(endpoint_name=model_endpoint)

    # Make predictions
    predictions = endpoint.predict(
        instances=[
            {"user_id": user_id}  # Include additional input features as required by your model
        ]
    )

    # Extract recommended items from the predictions
    recommended_items = [
        prediction["item_id"] for prediction in predictions.predictions if "item_id" in prediction
    ]

    print(f"Generated recommendations for user {user_id}: {recommended_items}")
    return recommended_items


if __name__ == "__main__":
    # Example usage

    # Training parameters
    training_data_path = "gs://your-bucket/path/to/training_data.csv"
    model_display_name = "baseball_recommendation_model"
    project_id = "your-gcp-project-id"
    location = "us-central1"
    staging_bucket = "gs://your-staging-bucket"

    # Train the model
    train_recommendation_model(
        training_data_path=training_data_path,
        model_display_name=model_display_name,
        project_id=project_id,
        location=location,
        staging_bucket=staging_bucket,
    )

    # Prediction parameters
    model_endpoint = "projects/your-project/locations/us-central1/endpoints/your-endpoint-id"
    user_id = "user123"

    # Generate recommendations
    recommendations = predict_recommendations(
        model_endpoint=model_endpoint,
        user_id=user_id,
        project_id=project_id,
        location=location,
    )
    print(f"Recommendations for user {user_id}: {recommendations}")
