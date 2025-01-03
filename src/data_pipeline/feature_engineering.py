import pandas as pd


def extract_player_features(input_file, output_file):
    """Function to extract relevant player features.

    Args:
        input_file: Path to the input file.
        output_file: Path to the output file.
    """
    df = pd.read_csv(input_file)

    # Calculate batting average (BA): hits / at-bats
    df['Batting_Average'] = df['Hits'] / df['At_Bats']

    # Calculate on-base percentage (OBP): (hits + walks + hit by pitch) / (at-bats + walks + hit by pitch + sacrifice flies)
    df['On_Base_Percentage'] = (df['Hits'] + df['Walks'] + df['Hit_By_Pitch']) / (
        df['At_Bats'] + df['Walks'] + df['Hit_By_Pitch'] + df['Sacrifice_Flies']
    )

    # Calculate slugging percentage (SLG): total bases / at-bats
    # Assuming columns for singles, doubles, triples, and home runs
    df['Total_Bases'] = (df['Singles'] + 2 * df['Doubles'] + 3 * df['Triples'] + 4 * df['Home_Runs'])
    df['Slugging_Percentage'] = df['Total_Bases'] / df['At_Bats']

    # Calculate ERA (earned run average) for pitchers: (earned runs * 9) / innings pitched
    df['ERA'] = (df['Earned_Runs'] * 9) / df['Innings_Pitched']

    # Calculate WHIP (walks and hits per inning pitched): (walks + hits) / innings pitched
    df['WHIP'] = (df['Walks_Allowed'] + df['Hits_Allowed']) / df['Innings_Pitched']

    # Other metrics can be added here based on additional available data

    # Save the processed DataFrame to the output file
    df.to_csv(output_file, index=False)


def extract_game_logs_features(input_file, output_file):
    """
    Function to extract relevant features from game logs.

    Args:
        input_file: Path to the input file.
        output_file: Path to the output file.
    """
    df = pd.read_csv(input_file)

    # Example metrics for game logs:
    # - Win percentage
    df['Win_Percentage'] = df['Wins'] / (df['Wins'] + df['Losses'])

    # - Average runs scored per game
    df['Average_Runs_Per_Game'] = df['Runs_Scored'] / df['Games_Played']

    # - Average runs allowed per game
    df['Average_Runs_Allowed_Per_Game'] = df['Runs_Allowed'] / df['Games_Played']

    # Save the processed DataFrame to the output file
    df.to_csv(output_file, index=False)


def extract_statcast_features(input_file, output_file):
    """
    Function to extract relevant features from Statcast data.

    Args:
        input_file: Path to the input file.
        output_file: Path to the output file.
    """
    df = pd.read_csv(input_file)

    # Example metrics for Statcast data:
    # - Average exit velocity
    df['Average_Exit_Velocity'] = df['Exit_Velocity'].mean()

    # - Average launch angle
    df['Average_Launch_Angle'] = df['Launch_Angle'].mean()

    # - Barrel rate: barrels / batted ball events
    df['Barrel_Rate'] = df['Barrels'] / df['Batted_Ball_Events']

    # Save the processed DataFrame to the output file
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    player_input_file = "data/raw/player_stats.csv"
    player_output_file = "data/processed/player_features.csv"
    extract_player_features(player_input_file, player_output_file)

    game_logs_input_file = "data/raw/game_logs.csv"
    game_logs_output_file = "data/processed/game_logs_features.csv"
    extract_game_logs_features(game_logs_input_file, game_logs_output_file)

    statcast_input_file = "data/raw/statcast.csv"
    statcast_output_file = "data/processed/statcast_features.csv"
    extract_statcast_features(statcast_input_file, statcast_output_file)
