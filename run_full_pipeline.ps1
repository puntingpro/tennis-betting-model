# A script to run the full data pipeline and train/evaluate the champion LightGBM model.

# Function to check for errors after each python script execution
function Check-Error {
    if ($LASTEXITCODE -ne 0) {
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "A critical error occurred. Halting the pipeline."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
    }
}

# --- Step 1: Prepare and Build Data ---
echo "--- Running Data Preparation ---"
python main.py prepare-data
Check-Error

echo "--- Creating Player Map ---"
python main.py create-player-map
Check-Error

echo "--- Building Features and Data Assets ---"
python main.py build
Check-Error


# --- Step 2: Train and Evaluate the LightGBM Model ---
echo "--- Training the LightGBM Model ---"
python main.py model
Check-Error

echo "--- Running the Backtest ---"
python main.py backtest realistic
Check-Error


# --- Step 3: Analyze Results ---
echo "--- Pipeline Complete ---"
echo "Launch the dashboard to analyze the results:"
python main.py dashboard
