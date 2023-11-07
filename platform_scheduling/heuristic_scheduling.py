# path/filename: test_scheduling_system.py

import random

# Constants for the simulation
NUM_DOCKS = 10  # Number of docks in the simulation
NUM_VEHICLES = 10  # Number of vehicles in the simulation
MAX_DOCK_VISITS = 10  # Maximum number of docks a vehicle may need to visit
LOADING_TIME = 30  # The time needed for loading or unloading at a dock in minutes

# Generate the initial docks and vehicles
docks = [{'id': f'Dock {i + 1}', 'next_available_time': 0} for i in range(NUM_DOCKS)]
vehicles = [{'id': f'Vehicle {i+1}',
             'required_docks': random.sample([dock['id'] for dock in docks], random.randint(1, MAX_DOCK_VISITS)),
             'next_available_time': 0} for i in range(NUM_VEHICLES)]


def find_next_available_time(vehicle, dock, global_schedules, loading_time):
    # Extract dock schedules specific to the dock we're examining
    dock_schedules = [s for s in global_schedules if s['dock_id'] == dock['id']]
    # Sort dock schedules by start time
    dock_schedules.sort(key=lambda x: x['start_time'])

    # Find the first time slot where the vehicle can start after its current availability
    start_time = vehicle['next_available_time']
    for schedule in dock_schedules:
        # If there is a gap between the vehicle's available time and the next scheduled time at the dock, use it
        if start_time + loading_time <= schedule['start_time']:
            break  # Found a slot that fits in the current gap
        # If the vehicle's start time is before the end time of the current schedule, push it back
        if start_time < schedule['end_time']:
            start_time = schedule['end_time']

    # Return the calculated start and end time
    end_time = start_time + loading_time
    return start_time, end_time

def generate_optimized_schedules(vehicles, docks, loading_time):
    global_schedules = []  # This will keep track of all dock schedules
    dock_sequences = {vehicle['id']: [] for vehicle in vehicles}  # Initialize dock sequences for each vehicle

    # Iterate over vehicles and their required docks to schedule them
    for vehicle in vehicles:
        for dock_id in vehicle['required_docks']:
            # Find the corresponding dock from the docks list
            dock = next((d for d in docks if d['id'] == dock_id), None)
            if not dock:
                continue  # If the dock is not found, skip to the next required dock

            # Use the find_next_available_time function to get the start and end time for the vehicle at the dock
            start_time, end_time = find_next_available_time(vehicle, dock, global_schedules, loading_time)

            # Schedule the vehicle for this dock
            global_schedules.append({
                'vehicle_id': vehicle['id'],
                'dock_id': dock['id'],
                'start_time': start_time,
                'end_time': end_time
            })

            # Update dock sequences
            dock_sequences[vehicle['id']].append(dock_id)

            # Update the vehicle's and dock's next available time
            vehicle['next_available_time'] = end_time
            dock['next_available_time'] = end_time

    return global_schedules, dock_sequences

# With the functions redefined, let's create the schedules
optimized_schedules, optimized_dock_sequences = generate_optimized_schedules(vehicles, docks, LOADING_TIME)

# Output the optimized schedules and sequences for verification
optimized_schedules, optimized_dock_sequences
