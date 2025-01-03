import pandas as pd


def clean_mlb_data(input_file, output_file):
    """
    Function to clean MLB datasets.

    Args:
        input_file: Path to the input file.
        output_file: Path to the output file.
    """
    df = pd.read_csv(input_file)

    # Handle missing values: Fill numerical columns with 0 and categorical columns with "Unknown"
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(0, inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna("Unknown", inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Convert data types: Convert dates to datetime and ensure numeric columns are properly typed
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    for col in df.select_dtypes(include='object').columns:
        if df[col].str.isnumeric().any():
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Perform any necessary data transformations: For example, standardizing column names
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    # Save cleaned DataFrame to the output file
    df.to_csv(output_file, index=False)


def clean_game_logs(input_file, output_file):
    """
    Function to clean game logs data.

    Args:
        input_file: Path to the input file.
        output_file: Path to the output file.
    """
    df = pd.read_csv(input_file)

    # Handle missing values
    df.fillna({
        'Wins': 0,
        'Losses': 0,
        'Runs_Scored': 0,
        'Runs_Allowed': 0,
        'Date': None
    }, inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Convert 'Date' column to datetime format
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Standardize column names
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    # Save cleaned DataFrame to the output file
    df.to_csv(output_file, index=False)


def clean_statcast_data(input_file, output_file):
    """
    Function to clean Statcast data.

    Args:
        input_file: Path to the input file.
        output_file: Path to the output file.
    """
    df = pd.read_csv(input_file)

    # Handle missing values: Replace NaNs with appropriate values
    df.fillna({
        'Exit_Velocity': df['Exit_Velocity'].mean(),
        'Launch_Angle': df['Launch_Angle'].mean(),
        'Barrels': 0
    }, inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Standardize column names
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    # Save cleaned DataFrame to the output file
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    # Clean player stats dataset
    player_input_file = "data/raw/player_stats.csv"
    player_output_file = "data/cleaned/player_stats_cleaned.csv"
    clean_mlb_data(player_input_file, player_output_file)

    # Clean game logs dataset
    game_logs_input_file = "data/raw/game_logs.csv"
    game_logs_output_file = "data/cleaned/game_logs_cleaned.csv"
    clean_game_logs(game_logs_input_file, game_logs_output_file)

    # Clean Statcast dataset
    statcast_input_file = "data/raw/statcast.csv"
    statcast_output_file = "data/cleaned/statcast_cleaned.csv"
    clean_statcast_data(statcast_input_file, statcast_output_file)
