# F1 Podium Prediction Model 🏎️

## Overview
This Python application predicts the probability of a Formula 1 driver securing a podium finish in an upcoming race. Using a comprehensive weighted scoring system that I developed through extensive research and testing, the model analyzes multiple factors that influence race outcomes.

[Watch the Demo](https://github.com/cris-mbici/podium_probability/raw/main/probability_demo.mp4)

## The Science Behind the Model

Working on this project taught me the importance of combining statistical analysis with domain knowledge. While I wrote all the code myself, I collaborated with Claude AI to develop the mathematical model and determine the appropriate weighting for each factor. The final prediction formula represents weeks of testing and refinement:

```
Podium_Probability = (0.22 * Grid_Score) + (0.14 * Constructor_Score) + 
                     (0.10 * Qualifying_Score) + (0.08 * Recent_Form_Score) + 
                     (0.07 * Team_Form_Score) + (0.06 * Podium_Rate_Score) + 
                     (0.05 * Win_Rate_Score) + (0.04 * Driver_Score) + 
                     (0.04 * Circuit_Performance_Score) + (0.20 * Other_Factors)
```

The data used to train and test this model comes from a comprehensive Formula 1 racing database (1950-2024) sourced from Kaggle.

## Key Features

✅ **Three-Component Scoring System** with detailed analysis of:
  - Constructor/Team Performance
  - Driver Performance 
  - Race-Specific Factors

✅ **Flexible Input Options** allowing for both manual score entry and automated calculation

✅ **Weather Impact Analysis** that considers driver performance in various conditions

✅ **Circuit-Specific Performance** evaluation based on historical results

✅ **Normalized Scoring System** that ensures all factors fall within a 0-1 range for consistent weighting

## What I Learned

### 1. **Data Normalization Techniques**
I discovered that normalizing all factors to a 0-1 scale was essential for creating a balanced model. For example, converting grid position (1-20) to a score where higher is better required this transformation:
```python
Grid_Score = (21 - Grid_Position) / 20
```
This ensures that pole position (grid=1) receives a score near 1.0, while starting last (grid=20) receives a score near 0.05.

### 2. **Weighted Formula Development**
Creating an effective prediction model required carefully balancing multiple factors. Through extensive testing, I learned that grid position (22%) and other unpredictable race-day factors (20%) have the most significant impact on podium chances.

### 3. **Function Modularization**
I structured my code into specialized functions to improve readability and maintainability:
```python
def calculate_constructor_score(constructor_data):
    # Code calculating team performance
    
def calculate_driver_score(driver_data):
    # Code evaluating driver quality
    
def calculate_other_factors(other_factor_data):
    # Code handling race-specific conditions
```
This approach made debugging much easier and allowed me to isolate and refine each component individually.

### 4. **Dictionary Data Organization**
I learned to use dictionaries to organize related data and pass it efficiently between functions:
```python
constructor_data = {
    'current_points': current_points,
    'max_constructor_points_in_season': max_points,
    # More data...
}
```
This keeps the code clean and makes it easier to understand what data each function needs.

### 5. **Input Validation**
Handling different types of input (floats, comma-separated lists, etc.) taught me the importance of robust input processing:
```python
last_three_points = input("Enter team's points in last three races (comma-separated): ")
last_three_points = list(map(float, last_three_points.split(",")))
```

### 6. **Statistical Analysis Application**
Converting raw F1 statistics into meaningful prediction factors required understanding performance trends and their relative importance.

## How to Use

1. Run the script: `python Podium_Chance.py`
2. Follow the prompts to enter race information
3. Choose between manual entry or guided calculation for:
   - Constructor/team performance
   - Driver performance
   - Race-specific factors
4. Review the calculated podium probability

## Sample Output

```
Your driver has a 68.75% chance of securing a podium
```

## Future Improvements

As I continue to develop my programming skills, I plan to enhance this project by:

- Creating a GUI interface using Tkinter or PyQt
- Implementing automatic data retrieval from F1 statistics APIs
- Adding a visualization component to show how each factor contributes to the final prediction
- Developing a machine learning approach to refine the formula weights based on historical accuracy

## Final Thoughts

This project represents my passion for both Formula 1 racing and data that tells a story. It demonstrates my ability to apply mathematical concepts to real-world problems and develop structured, modular code. I'm particularly proud of how I combined statistical analysis with domain knowledge to create a practical tool that produces meaningful predictions.
I used Claude to help me get the theoretical know-how while avoiding getting overwhelmed by technical information. I hope this project serves as a solid foundation to my future in data science!
