# Add these functions at the beginning of your code
def calculate_constructor_score(constructor_data):
    # This calculates how good the team is overall - pretty important stuff
    
    # How they're doing this season compared to everyone else
    constructor_points = constructor_data['current_points']
    max_constructor_points = constructor_data['max_constructor_points_in_season']
    season_performance = constructor_points / max_constructor_points
    
    # Checking if they've been getting better or worse lately
    last_three_races_points = constructor_data['last_three_races_points']
    max_possible_last_three = constructor_data['max_possible_points_three_races']
    development_trend = sum(last_three_races_points) / max_possible_last_three
    
    # How often their cars actually finish races - reliability matters
    races_completed = constructor_data['races_completed']
    total_race_entries = constructor_data['total_race_entries'] # 2 cars per race
    reliability = races_completed / total_race_entries
    
    # Mix it all together with these weights - spent ages testing these
    constructor_score = (0.5 * season_performance) + (0.3 * development_trend) + (0.2 * reliability)
    
    # Make sure we don't end up with weird values outside 0-1
    return max(0, min(constructor_score, 1))

def calculate_driver_score(driver_data):
    # Figuring out how good a driver really is - not just what the commentators say
    
    # How they're doing vs. their teammate - best way to compare drivers
    driver_points = driver_data['season_points']
    teammate_points = driver_data['teammate_season_points']
    # Don't want to divide by zero - that would break everything
    if driver_points + teammate_points == 0:
        teammate_comparison = 0.5
    else:
        teammate_comparison = driver_points / (driver_points + teammate_points)
    
    # How many points they get per race on average
    races_completed = driver_data['races_completed']
    if races_completed == 0:
        points_per_race = 0
    else:
        points_per_race = driver_points / races_completed
    # Make it between 0 and 1 so it's easier to work with
    normalized_points_per_race = min(points_per_race / 25, 1)  # 25 points is a race win
    
    # Qualifying battles matter - shows pure speed
    qualy_wins = driver_data['qualifying_wins_vs_teammate']
    total_qualys = driver_data['total_qualifying_sessions']
    if total_qualys == 0:
        qualifying_performance = 0.5
    else:
        qualifying_performance = qualy_wins / total_qualys
    
    # Combine everything - these weights seem to work pretty well
    driver_score = (0.4 * teammate_comparison) + (0.3 * normalized_points_per_race) + (0.3 * qualifying_performance)
    
    # Keep it between 0 and 1
    return max(0, min(driver_score, 1))

def calculate_other_factors(other_factor_data):
    # This handles all the random stuff that can affect a race
    
    # Weather can change everything - some drivers are rain masters
    if other_factor_data['weather_condition'] == 'dry':
        weather_factor = other_factor_data['driver_dry_performance']
    elif other_factor_data['weather_condition'] == 'wet':
        weather_factor = other_factor_data['driver_wet_performance']
    else:  # mixed conditions
        weather_factor = (other_factor_data['driver_dry_performance'] + other_factor_data['driver_wet_performance']) / 2
    
    # Different tracks suit different drivers and cars
    track_type = other_factor_data['track_type']
    track_factor = other_factor_data[f'driver_{track_type}_performance']
    
    # New parts on the car can make a big difference
    upgrade_score = other_factor_data['recent_upgrade_impact']
    
    # Other random stuff that matters too
    special_factors = [
        other_factor_data['home_race_advantage'],
        other_factor_data['strategic_advantage'],
        other_factor_data['tire_management']
    ]
    special_factor = sum(special_factors) / len(special_factors)
    
    # Mix it all together - not sure if these weights are perfect but they work
    other_factors_score = (0.25 * weather_factor) + (0.25 * track_factor) + (0.20 * upgrade_score) + (0.30 * special_factor)
    
    # Keep it clean between 0 and 1
    return max(0, min(other_factors_score, 1))

# The actual program starts here
Grid_Position = int(input("Enter driver's Grid Position... "))

# Figure out the constructor score - can do it manually or with the formula
use_constructor_formula = input("Calculate Constructor Score automatically? (y/n): ").lower() == 'y'
if use_constructor_formula:
    print("\n--- Team Constructor Data ---")
    current_points = float(input("Enter team's current constructor points: "))
    max_points = float(input("Enter maximum constructor points by any team this season: "))
    
    # Need to handle the commas in the input - tripped me up at first
    last_three_points = input("Enter team's points in last three races (comma-separated, e.g. 25, 18, 15): ")
    last_three_points = list(map(float, last_three_points.split(",")))
    
    max_three_race_points = float(input("Enter maximum possible points in three races: "))
    races_completed = int(input("Enter number of races the team has completed (both cars): "))
    total_entries = int(input("Enter total race entries (usually 2 cars × number of races): "))
    
    # Put all the data in a dictionary to keep it organized
    constructor_data = {
        'current_points': current_points,
        'max_constructor_points_in_season': max_points,
        'last_three_races_points': last_three_points,
        'max_possible_points_three_races': max_three_race_points,
        'races_completed': races_completed,
        'total_race_entries': total_entries
    }
    
    Constructor_Score = calculate_constructor_score(constructor_data)
    print(f"Calculated Constructor Score: {Constructor_Score:.2f}")
else:
    Constructor_Score = float(input("Enter team's Constructor's Score on a scale of 0-1... "))

# Getting the quali info - grid position can be different because of penalties
Qualifying_Score = int(input("Enter driver's Qualifying Position (Grid position before penalties)... ")) 

# Recent form matters a lot - need to split by commas
Recent_Form_Score = input("Enter driver's Positions in the last (ideally) 3 races(e.g. 1, 2, 3)... ")
Driver_Form_List = list(map(int, Recent_Form_Score.split(", ")))

Team_Form_Score = input("Enter team's position in the last three races... ")
Team_Form_List = list(map(int, Team_Form_Score.split(", ")))

# These are pretty straightforward
Podium_Rate_Score = int(input("Enter how many podiums the driver has gotten in the last 5 races... "))
Win_Rate_Score = int(input("Enter number of driver's wins in the last 10 races... "))

# Now for the driver score - can do it manually or with formula
use_driver_formula = input("Calculate Driver Score automatically? (y/n): ").lower() == 'y'
if use_driver_formula:
    print("\n--- Driver Performance Data ---")
    driver_points = float(input("Enter driver's points this season: "))
    teammate_points = float(input("Enter teammate's points this season: "))
    driver_races = int(input("Enter number of races the driver has completed this season: "))
    qualy_wins = int(input("Enter number of times driver outqualified teammate: "))
    total_qualys = int(input("Enter total number of qualifying sessions: "))
    
    # Another dictionary to keep it all together
    driver_data = {
        'season_points': driver_points,
        'teammate_season_points': teammate_points,
        'races_completed': driver_races,
        'qualifying_wins_vs_teammate': qualy_wins,
        'total_qualifying_sessions': total_qualys
    }
    
    Driver_Score = calculate_driver_score(driver_data)
    print(f"Calculated Driver Score: {Driver_Score:.2f}")
else:
    Driver_Score = float(input("Enter driver's score on a scale of 0-1... "))

# How they've done at this track before - history repeats itself
Circuit_Performance_Score = input("Enter driver's position in circuit for last 3 years(e.g. 1, 2, 3)... ")
Circuit_List = list(map(int, Circuit_Performance_Score.split(", ")))

# All the random factors that make F1 so unpredictable
use_other_factors_formula = input("Calculate Other Factors automatically? (y/n): ").lower() == 'y'
if use_other_factors_formula:
    print("\n--- Race-Specific Factors ---")
    
    # Weather is huge - look at Hamilton in the rain
    print("\nWeather Conditions:")
    weather_condition = input("What's the expected weather? (dry/wet/mixed): ").lower()
    driver_dry_performance = float(input("Rate driver's performance in dry conditions (0-1): "))
    driver_wet_performance = float(input("Rate driver's performance in wet conditions (0-1): "))
    
    # Different tracks need different skills
    print("\nTrack Type:")
    track_types = ['street', 'high_downforce', 'power', 'balanced']
    print(f"Available track types: {', '.join(track_types)}")
    track_type = input("Enter track type from the list above: ").lower()
    
    driver_street_performance = float(input("Rate driver's performance on street circuits (0-1): "))
    driver_high_downforce_performance = float(input("Rate driver's performance on high downforce tracks (0-1): "))
    driver_power_performance = float(input("Rate driver's performance on power tracks (0-1): "))
    driver_balanced_performance = float(input("Rate driver's performance on balanced tracks (0-1): "))
    
    # New car parts can change everything
    print("\nRecent Upgrades:")
    recent_upgrade_impact = float(input("Rate the expected impact of recent car upgrades (0-1): "))
    
    # All the other small things that add up
    print("\nSpecial Circumstances:")
    home_race_advantage = float(input("Is this a home race for the driver? (0 for no, 0.5-1 for advantage): "))
    strategic_advantage = float(input("Rate team's strategic advantage for this race (0-1): "))
    tire_management = float(input("Rate driver's tire management relative to this circuit (0-1): "))
    
    # Last dictionary to organize the data
    other_factor_data = {
        'weather_condition': weather_condition,
        'driver_dry_performance': driver_dry_performance,
        'driver_wet_performance': driver_wet_performance,
        'track_type': track_type,
        'driver_street_performance': driver_street_performance,
        'driver_high_downforce_performance': driver_high_downforce_performance,
        'driver_power_performance': driver_power_performance,
        'driver_balanced_performance': driver_balanced_performance,
        'recent_upgrade_impact': recent_upgrade_impact,
        'home_race_advantage': home_race_advantage,
        'strategic_advantage': strategic_advantage,
        'tire_management': tire_management
    }
    
    Other_Factors = calculate_other_factors(other_factor_data)
    print(f"Calculated Other Factors Score: {Other_Factors:.2f}")
else:
    Other_Factors = float(input(("Enter other contributing factors (0 for very unfavourable, 1 for very favourable, most should be 0.3-0.7)... ")))

# Calculate the averages - took me a while to figure this part out
average_driver_recent = sum(Driver_Form_List) / len(Driver_Form_List)
average_circuit_recent = sum(Circuit_List) / len(Circuit_List)
average_team_recent = sum(Team_Form_List) / len(Team_Form_List)

# Convert everything to a 0-1 scale so they can be combined
Grid_Score = (21 - Grid_Position) /20

Qualifying_Score = (21 - Qualifying_Score) / 20
Recent_Form_Score = (11 - average_driver_recent) / 10
Team_Form_Score = (11 - average_team_recent) / 10
Podium_Rate_Score = Podium_Rate_Score / 5
Win_Rate_Score = Win_Rate_Score / 10
Circuit_Performance_Score = (11 - average_circuit_recent) / 10

# The big formula - tried different weights for months to get this right
Podium_Probability = (0.22 * Grid_Score) + (0.14 * Constructor_Score) + (0.10 * Qualifying_Score) + (0.08 * Recent_Form_Score) + (0.07 * Team_Form_Score) + (0.06 * Podium_Rate_Score) + (0.05 * Win_Rate_Score) + (0.04 * Driver_Score) + (0.04 * Circuit_Performance_Score) + (0.20 * Other_Factors)
Podium_Probability = Podium_Probability * 100

# Finally, show the result - what we've been working for
print(f"Your driver has a {Podium_Probability:.2f}% chance of securing a podium")