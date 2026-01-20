# Beta Band ERD in Motor Prediction

"""
IMPORTS
"""
import os
from psychopy import prefs, logging

#disable the frame drop warning limit - report ALL dropped frames
logging.console.setLevel(logging.WARNING)  #keep warnings visible

from psychopy import visual, core, event, gui, monitors
import random
import itertools
import pandas as pd
import numpy as np
import math
import pyautogui
from datetime import datetime

from pylsl import StreamInfo, StreamOutlet, local_clock #lsl software setup
from human_curve_generator import HumanizeMouseTrajectory #for human-like movement

#get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

#data directory (relative to script location)
DATA_DIR = os.path.join(SCRIPT_DIR, 'experiment_data')

# ===== DEBUG SETTINGS =====
#set to False for actual data collection (slows down timing)
DEBUG_MODE = False 
# ==========================

"""
LSL STREAM SETUP
"""
#Setting up LSL outlet for EEG markers
info = StreamInfo(name = "EEG_Markers", type = "Markers", channel_count = 1, channel_format = "string", source_id = "marker_001")
outlet = StreamOutlet(info)

"""
MONITOR AND WINDOW SETUP
"""
MON_DISTANCE = 60  #cm between subjects' eyes and the monitor (generally recommended for health 50-76)
MON_WIDTH = 29  #width of monitor in cm (adjust to the computer used for the experiment)

logical_screen_size = pyautogui.size()  # This gives scaled size: 1470 x 956

#set MON_SIZE to match what psychopy will actually use (double scaling due to Retina display)
MON_SIZE = [logical_screen_size[0] * 2, logical_screen_size[1] * 2]  # [2940, 1912]

if DEBUG_MODE:
    print(f"Logical screen size: {logical_screen_size}")
    print(f"Setting MON_SIZE to: {MON_SIZE}")

MON_WH_PROPORTION = MON_SIZE[0] / MON_SIZE[1]

NOMINAL_FRAME_RATE = 120 #hz
SAVE_FOLDER = 'uncertainty_motor_data' 

#Create monitor object (for visual angle calculations)
my_monitor = monitors.Monitor('testMonitor', width=MON_WIDTH, distance=MON_DISTANCE)  #creating monitor object (to control size of the stim in degrees)
my_monitor.setSizePix(MON_SIZE)

#Participant dialog window
subj_info = {'ID': '', 
             'age': '', 
             'gender': ['woman', 'man', 'non-binary', 'prefer not to say'], 
             'native language': '', 
             'handedness': ['right', 'left', 'ambidextrous']}
dlg = gui.DlgFromDict(dictionary=subj_info, title='EEG Experiment')
if not dlg.OK:
    core.quit()

#Create psychopy window
win = visual.Window(monitor = my_monitor, units = 'deg', fullscr = True, allowGUI = False, color = 'black', waitBlanking=True)  #initiating psychopy window, using degrees as units

win.flip() #to force psychopy to finish initialising graphics
core.wait(0.5)

#Initlialise mouse object
mouse = event.Mouse(win = win)
mouse.setVisible(False)

win.mouseVisible = False #hide system cursor

#Store screen scaling information globally
SCREEN_SCALE_X = 2.0  #Retina display scaling (default)
SCREEN_SCALE_Y = 2.0

def get_actual_screen_info():
    """Gets screen size and calculates scaling factors for Retina displays."""
    global SCREEN_SCALE_X, SCREEN_SCALE_Y
    
    try:
        # Get PyAutoGUI screen size (logical coordinates)
        pyautogui_size = pyautogui.size()
        print(f"PyAutoGUI reports screen size (logical): {pyautogui_size}")
        
        # Get psychopy window size (physical/internal coordinates)
        psychopy_size = win.size
        print(f"PsychoPy window size (internal): {psychopy_size}")
        
        # Calculate scaling factor
        SCREEN_SCALE_X = psychopy_size[0] / pyautogui_size[0]
        SCREEN_SCALE_Y = psychopy_size[1] / pyautogui_size[1]
        
        print(f"Detected scaling factors: X={SCREEN_SCALE_X:.2f}, Y={SCREEN_SCALE_Y:.2f}")
        
        return pyautogui_size, (SCREEN_SCALE_X, SCREEN_SCALE_Y)
        
    except Exception as e:
        print(f"Error getting screen info: {e}")
        return win.size, (1.0, 1.0)


def deg2pix(win, pos):
    """Converts position from degrees of visual angle to pixels."""
    x_deg, y_deg = pos
    width, height = win.size  #psychopy internal pixels
    
    mon = win.monitor
    dist = float(mon.getDistance())
    scr_width = float(mon.getWidth())
    scr_res = mon.getSizePix()
    
    pix_per_cm = float(scr_res[0]) / scr_width
    cm_per_deg = dist * math.tan(math.radians(1))
    pix_per_deg = pix_per_cm * cm_per_deg
    
    #Convert degrees to pixels in psychopy's coordinate system
    x_pix = width/2 + x_deg * pix_per_deg
    y_pix = height/2 - y_deg * pix_per_deg
    
    return (x_pix, y_pix)
    

def pix2deg(win, pos_pix):
    """Converts position from pixels to degrees of visual angle."""
    x_pix, y_pix = pos_pix
    width, height = win.size  #psychopy internal pixels
    
    mon = win.monitor
    dist = float(mon.getDistance())
    scr_width = float(mon.getWidth())
    scr_res = mon.getSizePix()
    
    pix_per_cm = float(scr_res[0]) / scr_width
    cm_per_deg = dist * math.tan(math.radians(1))
    pix_per_deg = pix_per_cm * cm_per_deg
    
    #convert pixels to degrees in psychopy's coordinate system
    x_deg = (x_pix - width / 2) / pix_per_deg
    y_deg = (height / 2 - y_pix) / pix_per_deg
    
    return (x_deg, y_deg)


"""
TIMING SETUP
"""

#Frame measurement
measured = None

#default frame_period
frame_period = 1.0 / float(NOMINAL_FRAME_RATE)

try:
    #measuring frame rate
    measured = win.getActualFrameRate(nIdentical=10, nMaxFrames=100, nWarmUpFrames=10, threshold=1) #requirement for identical frame dur
    if DEBUG_MODE:
        print(f"Measured frame rate: {measured} Hz")
except Exception as e:
    if DEBUG_MODE:
        print(f"Could not measure frame rate: {e}")
    measured = None

#estimation of frame rate from flip times
if not measured or measured <= 0:
    if DEBUG_MODE:
        print("Measuring actual frame duration from flips...")
    test_stim = visual.TextStim(win, "+")
    test_clock = core.Clock()
    flip_times = []
    for i in range(30):
        start = test_clock.getTime()
        test_stim.draw()
        win.flip()
        flip_times.append(test_clock.getTime() - start)
    
    avg_flip = np.mean(flip_times)
    measured = 1.0 / avg_flip
    if DEBUG_MODE:
        print(f"Estimated frame rate from flips: {measured:.2f} Hz")

if measured and measured > 0:
    frame_rate_Hz = measured
    if DEBUG_MODE:
        print(f"Using measured frame rate: {frame_rate_Hz:.2f} Hz")
    
    if abs(measured - NOMINAL_FRAME_RATE) > 10:
        if DEBUG_MODE:
            print(f"⚠️  NOTE: Measured rate ({measured:.1f} Hz) differs from nominal ({NOMINAL_FRAME_RATE} Hz)")
            print(f"   Using actual measured rate for accurate timing.")
else:
    #fall back to nominal if measurement failed
    frame_rate_Hz = NOMINAL_FRAME_RATE
    if DEBUG_MODE:
        print(f"⚠️  WARNING: Could not measure frame rate, using nominal: {frame_rate_Hz} Hz")

frame_dur = 1.0 / frame_rate_Hz
if DEBUG_MODE:
    print(f"Frame duration set to: {frame_dur*1000:.2f} ms")

#convert seconds to integer frames using measured refresh
def secs_to_frames(sec):
    return int(round(sec * frame_rate_Hz))

#convert frames to seconds
def frames_to_secs(frames):
    return frames * frame_dur

#Clock and timer
clock = core.Clock()  #used to time events on a trial-per-trial basis

#timing validation
def validate_timing():
    """Test and report actual timing performance."""
    print("\n" + "="*50)
    print("=== TIMING VALIDATION ===")
    print(f"Target frame rate: {NOMINAL_FRAME_RATE} Hz")
    print(f"Measured frame rate: {frame_rate_Hz:.3f} Hz")
    print(f"Frame duration: {frame_dur:.6f} s ({frame_dur*1000:.2f} ms)")
    
    #check if measured rate matches target
    if abs(frame_rate_Hz - NOMINAL_FRAME_RATE) > 5:
        print(f"⚠️  WARNING: Measured rate differs from target by {abs(frame_rate_Hz - NOMINAL_FRAME_RATE):.1f} Hz")
    
    #test actual flip timing
    print("\nTesting frame flip consistency...")
    test_stim = visual.TextStim(win, "+")
    flip_times = []
    
    for i in range(30):  # Test 30 frames
        start = clock.getTime()
        test_stim.draw()
        win.flip()
        flip_times.append(clock.getTime() - start)
    
    avg_flip = np.mean(flip_times)
    std_flip = np.std(flip_times)
    max_flip = np.max(flip_times)
    min_flip = np.min(flip_times)
    
    print(f"Average flip time: {avg_flip*1000:.2f} ms")
    print(f"Std deviation: {std_flip*1000:.2f} ms")
    print(f"Range: [{min_flip*1000:.2f}, {max_flip*1000:.2f}] ms")
    print(f"Expected: ~{frame_dur*1000:.2f} ms")
    
    #check for dropped frames
    dropped_frames = sum(1 for t in flip_times if t > frame_dur * 1.5)
    if dropped_frames > 0:
        print(f"⚠️  WARNING: {dropped_frames}/30 frames took longer than expected")
    else:
        print("✓ Frame timing is consistent")
    
    print("="*50 + "\n")
        
if DEBUG_MODE:
    validate_timing()

"""
SAFETY CHECK
"""
def check_escape():
    """Checks if escape key was pressed and quits the experiment if so."""
    try:
        if 'escape' in event.getKeys():
            print("Experiment terminated by user.")
            win.close()
            core.quit()
    except (AttributeError, TypeError):
        pass

def time_based_wait(duration_sec, check_interval=0.001):
    """
    Waits for a specified duration while periodically checking for escape key.
    Used when no visual stimuli are displayed.
    """
    start_time = clock.getTime()
    target_time = start_time + duration_sec
    
    #use a hybrid approach: mostly precise wait, with periodic escape checks
    remaining = duration_sec
    while remaining > check_interval:
        check_escape()
        #wait most of remaining time, but leave time for next escape check
        wait_chunk = min(0.05, remaining - check_interval)  #wait in 50 ms chunks max
        core.wait(wait_chunk, hogCPUperiod=0)
        remaining = target_time - clock.getTime()
    
    #final precise wait to hit exact target time (no escape check in last ms)
    remaining = target_time - clock.getTime()
    if remaining > 0:
        core.wait(remaining, hogCPUperiod=0)
        
def safe_wait(frames, **cue_kwargs):
    """
    Waits for specified number of frames while displaying stimuli and checking for escape key.
    Uses time-based duration control to prevent timing issues from dropped frames.
    """
    frames = int(frames)
    if frames <= 0:
        return
    
    start_time = clock.getTime()
    expected_duration = frames * frame_dur
    target_time = start_time + expected_duration
    
    frame_count = 0
    while clock.getTime() < target_time:
        check_escape()
        draw_cues(win, **cue_kwargs)
        
        win.flip()
        frame_count += 1
        
        #safety check to prevent infinite loops
        if frame_count > frames * 2:  
            print(f"⚠️  safe_wait: drew {frame_count} frames (expected {frames}), exiting")
            break
    
    end_time = clock.getTime()
    actual_duration = end_time - start_time
    timing_error = actual_duration - expected_duration
    
    #log timing precision only in DEBUG mode 
    if DEBUG_MODE:
        print(f"safe_wait: {frames} frames, expected {expected_duration*1000:.1f}ms, "
              f"actual {actual_duration*1000:.1f}ms, error {timing_error*1000:+.1f}ms, "
              f"drew {frame_count} frames")

"""
STIMULI DEFINITIONS
"""
#Fixation cross
fix_cross = visual.TextStim(win, '+')

#img stimuli scale factor
scale_factor = 0.3

#image stim representing partner (relative to script directory)
img_partner_social = os.path.join(SCRIPT_DIR, "partner.JPG")
img_partner_non_social = os.path.join(SCRIPT_DIR, "average_blurred_partner.JPG")
partner_img = visual.ImageStim(win, image = img_partner_social, pos = (0, 3.1), size = None)
partner_img.size = partner_img.size * scale_factor

img_social_hand = os.path.join(SCRIPT_DIR, "hand_icon_left.png")
social_hand = visual.ImageStim(win, image = img_social_hand, pos = (-2, 3.1), size = None)
social_hand.size = social_hand.size * scale_factor

partner_circle = visual.Circle(win, radius = 0.5, pos = social_hand.pos, lineColor = 'grey', fillColor = 'grey')

#image symbolising cooperation/ observation condition (relative to script directory)
img_go = os.path.join(SCRIPT_DIR, "hand_icon_right.png")
img_nogo = os.path.join(SCRIPT_DIR, "hand_icon_nogo.png")
action = visual.ImageStim(win, image = img_go, pos = (2, 3.1), size = None)
action.size = action.size * scale_factor

#object to be moved
moving_object_pos = (social_hand.pos[0], social_hand.pos[1]) #starting position of the object
moving_object = visual.Circle(win, radius = 0.15, fillColor = 'red', lineColor = 'red', pos = moving_object_pos, draggable = True)

#participants starting location indicator 
participant_loc = visual.Circle(win, radius = 0.5, pos = (0, -3.1), fillColor = "grey", lineColor = "grey")

#possible target locations
def draw_location_circles(win, num_circles = 5, radius = 0.35, spacing = 3, y_pos = 0, n_green = 1):
    n_green = max(0, min(num_circles, n_green)) #safety check -> so the n_green is within the possible boundaries
    #randomised choice of indexes for green circles
    green_indices = np.random.choice(num_circles, size=n_green, replace=False).tolist()
    total_width = (num_circles - 1) * spacing
    start_x = -total_width / 2
    circles_info = []
    
    for i in range(num_circles):
        x_pos = start_x + i * spacing
        color = 'green' if i in green_indices else 'grey' #picking colour = green if the indice matches the list of chosen indices for green circles
        circle = visual.Circle(win, radius=radius, pos=(x_pos, y_pos), fillColor=color, lineColor=color)
        circles_info.append({'pos': (x_pos, y_pos), 'color': color, 'index': i}) #appending colour, coordinates, and index of drawn circles
    
    #return both circles_info and green_indices (positions numbered 0-4 from left to right)
    return circles_info, sorted(green_indices)

#visual cursor
visual_cursor = visual.Circle(win, radius = 0.1, fillColor = 'white', lineColor = 'white')

def get_corrected_mouse_pos(mouse):
    """
    Gets mouse position corrected for Retina display scaling.
    On Retina displays, positions are scaled 2x, so we need to correct this.
    """
    #get raw position from psychopy mouse (in degrees 2x scaled)
    raw_pos = mouse.getPos()

    #correct for scaling
    corrected_x = raw_pos[0] / SCREEN_SCALE_X
    corrected_y = raw_pos[1] / SCREEN_SCALE_Y
    
    return (corrected_x, corrected_y)

def update_visual_cursor(mouse):
    """Updates the visual cursor position to match the corrected mouse position."""
    visual_cursor.pos = get_corrected_mouse_pos(mouse)


def draw_cues(win, fix_cross = None, partner=None, social_hand = None, moving_object=None, participant_loc=None, action=None, circles_info=None, extra_stimuli=None, mouse = None, show_cursor = False):
    """Draws all relevant stimuli."""
    
    if fix_cross:
        fix_cross.draw()
    
    if circles_info:
        for circle_stim in circles_info:
            circle_stim.draw()
    
    if participant_loc:
        participant_loc.draw()
    if partner:
        partner.draw()
    if social_hand:
        social_hand.draw()
    if action:
        action.draw()
    if moving_object:
        moving_object.draw()
    if extra_stimuli:
        for stim in extra_stimuli:
            stim.draw()
    
    if show_cursor and mouse:
        update_visual_cursor(mouse)
        visual_cursor.draw()

#initialize screen scaling information
if DEBUG_MODE:  
    print("\n=== INITIALIZING SCREEN SCALING ===")
    get_actual_screen_info()
    
    #test mouse position correction
    test_pos_raw = mouse.getPos()
    test_pos_corrected = get_corrected_mouse_pos(mouse)
    print(f"Raw mouse position: {test_pos_raw}")
    print(f"Corrected position: {test_pos_corrected}")
    correction_factor = test_pos_raw[0]/test_pos_corrected[0] if test_pos_corrected[0] != 0 else None
    if correction_factor:
        print(f"Correction factor: {correction_factor:.2f}x")
    else:
        print("Correction factor: N/A")

def verify_monitor_calibration():
    """Checks if monitor parameters are correctly set and warns if they seem off."""
    if DEBUG_MODE:
        print("\n=== MONITOR CALIBRATION CHECK ===")
    
    mon = win.monitor
    dist = mon.getDistance()
    width_cm = mon.getWidth()
    size_pix = mon.getSizePix()
    
    # Calculate viewing angle
    half_width_cm = width_cm / 2
    half_angle_rad = math.atan(half_width_cm / dist)
    half_angle_deg = math.degrees(half_angle_rad)
    total_fov_deg = 2 * half_angle_deg
    
    if DEBUG_MODE:
        print(f"Monitor physical width: {width_cm} cm")
        print(f"Viewing distance: {dist} cm")
        print(f"Resolution: {size_pix}")
        print(f"Field of view: {total_fov_deg:.1f} degrees")
        print(f"Pixels per cm: {size_pix[0] / width_cm:.2f}")
    
    # Check if values are reasonable
    if total_fov_deg > 60 and DEBUG_MODE:
        print("⚠️  WARNING: Field of view seems too large (>60°)")
        print("   This can cause coordinate mapping issues.")
        print(f"   Consider increasing MON_WIDTH to ~{width_cm * 1.5:.1f} cm")
    
    if total_fov_deg < 20 and DEBUG_MODE:
        print("⚠️  WARNING: Field of view seems too small (<20°)")
        print("   This can cause coordinate mapping issues.")
        print(f"   Consider decreasing MON_WIDTH to ~{width_cm * 0.7:.1f} cm")
    
    # Calculate what 1 degree should be in pixels
    pix_per_cm = size_pix[0] / width_cm
    cm_per_deg = dist * math.tan(math.radians(1))
    pix_per_deg = pix_per_cm * cm_per_deg
    
    if DEBUG_MODE:
        print(f"Pixels per cm: {pix_per_cm:.2f}")   
        print(f"Cm per degree of visual angle: {cm_per_deg:.3f}")
        print(f"Pixels per degree: {pix_per_deg:.2f}")
        print(f"A 1° object should be ~{pix_per_deg:.0f} pixels or ~{cm_per_deg:.2f} cm on screen")
        print("="*50)

verify_monitor_calibration()

"""
FACTORIAL DESIGN (2 x 3 x 2) - BLOCK STRUCTURE
"""
#factors for the design
social_levels = ["social", "non_social"]
uncertainty_levels = ["certain", "low", "high"]
interaction_levels = ["go", "no_go"]

#trial and cue timing
fix_baseline = secs_to_frames(2.5) #2.5 sec fixation baseline
fix_jitter = [0, 0.2, 0.4, 0.6, 0.8, 1.0] #jitter options in sec

blank_frames = secs_to_frames(0.5) #0.5 sec blank screen pre cue


delays_sec = [0.25, 0.5]  #sec before nogo object retrieval
durations_sec = [1.0, 1.5]  #sec for cue presentation

delays = [secs_to_frames(s) for s in delays_sec] #frames before and after cue
durations = [secs_to_frames(s) for s in durations_sec] #frames for cue presenation

def randomize_without_consecutive_repeats(trial_list, key_func):
    """
    Randomizes trial list while ensuring no consecutive trials have the same condition.
    Uses a key function to determine what counts as a "repeat".
    """
    if len(trial_list) <= 1:
        return trial_list
    
    max_attempts = 1000
    for attempt in range(max_attempts):
        random.shuffle(trial_list)
        
        # Check if there are any consecutive repeats
        has_consecutive = False
        for i in range(len(trial_list) - 1):
            if key_func(trial_list[i]) == key_func(trial_list[i + 1]):
                has_consecutive = True
                break
        
        if not has_consecutive:
            return trial_list
    
    # If we can't find a perfect solution, return the best attempt
    if DEBUG_MODE:
        print(f"Warning: Could not find perfect randomization without consecutive repeats after {max_attempts} attempts")
    return trial_list


def make_block_trial_list(social_condition, trials_per_block=30):
    """
    Creates a list of trials for one block (either social or non-social).
    Each block contains all combinations of uncertainty and interaction levels.
    """
    # Generate all possible combinations for this social condition
    # 3 uncertainty levels x 2 interaction levels = 6 unique conditions
    combinations = list(itertools.product([social_condition], uncertainty_levels, interaction_levels))
    
    trial_list = []
    
    # Create trials by repeating the 6 conditions to reach trials_per_block
    repetitions_needed = trials_per_block // len(combinations)
    remainder = trials_per_block % len(combinations)
    
    # Add full repetitions
    for _ in range(repetitions_needed):
        for combo in combinations:
            social, uncertainty, interaction = combo
            
            # Random pre/post stim delays and durations
            delay_after = random.choice(delays)
            dur = random.choice(durations)
            
            # Random fixation duration with jitter for THIS trial
            fix_frames = fix_baseline + secs_to_frames(random.choice(fix_jitter))
            
            # Define the rule for retrieval vs observe
            task = "retrieve" if interaction == "go" else "observe"
            
            # EEG trigger codes
            trig_id = {
                'social': 1,
                'non_social': 0
            }[social] + {
                'certain': 1,
                'low': 2,
                'high': 5
            }[uncertainty] + (0 if interaction == 'no_go' else 1)
            
            trial_list.append({
                'social': social,
                'uncertainty': uncertainty,
                'interaction': interaction,
                'task': task,
                'lsl_trigger': trig_id,
                'fix_dur': fix_frames,
                'blank_dur': blank_frames,
                'delay_after': delay_after,
                'duration': dur
            })
    
    # Randomize without consecutive repeats
    # Check condition based on uncertainty and interaction combination
    trial_list = randomize_without_consecutive_repeats(
        trial_list, 
        key_func=lambda t: (t['uncertainty'], t['interaction'])
    )
    
    # Add trial numbers within block
    for i, trial in enumerate(trial_list):
        trial['trial_n_in_block'] = i + 1
    
    return trial_list


def make_experiment_blocks(num_blocks=8, trials_per_block=30):
    """
    Creates all experiment blocks with randomized order (alternating social/non-social).
    Ensures equal numbers of social and non-social blocks.
    """
    if num_blocks % 2 != 0:
        raise ValueError("Number of blocks must be even to have equal social/non-social blocks")
    
    # Randomly choose starting block type, then alternate
    start_with_social = random.choice([True, False])
    
    if start_with_social:
        block_conditions = ['social', 'non_social'] * (num_blocks // 2)
    else:
        block_conditions = ['non_social', 'social'] * (num_blocks // 2)
    
    # Create trials for each block
    blocks = []
    for block_num, social_condition in enumerate(block_conditions, start=1):
        block_trials = make_block_trial_list(social_condition, trials_per_block)
        
        # Add block number to each trial
        for trial in block_trials:
            trial['block_n'] = block_num
        
        blocks.append({
            'block_n': block_num,
            'social_condition': social_condition,
            'trials': block_trials
        })
    
    return blocks


def make_practice_block():
    """
    Creates a practice block with one trial of each of the 12 experimental conditions.
    This helps participants familiarize themselves with all condition types.
    """
    # Generate all 12 unique combinations
    combinations = list(itertools.product(social_levels, uncertainty_levels, interaction_levels))
    
    trial_list = []
    
    for combo in combinations:
        social, uncertainty, interaction = combo
        
        # Random pre/post stim delays and durations
        delay_after = random.choice(delays)
        dur = random.choice(durations)
        
        # Random fixation duration with jitter for THIS trial
        fix_frames = fix_baseline + secs_to_frames(random.choice(fix_jitter))
        
        # Define the rule for retrieval vs observe
        task = "retrieve" if interaction == "go" else "observe"
        
        # EEG trigger codes
        trig_id = {
            'social': 100,
            'non_social': 000
        }[social] + {
            'certain': 10,
            'low': 20,
            'high': 50
        }[uncertainty] + (0 if interaction == 'no_go' else 1)
        
        trial_list.append({
            'social': social,
            'uncertainty': uncertainty,
            'interaction': interaction,
            'task': task,
            'lsl_trigger': trig_id,
            'fix_dur': fix_frames,
            'blank_dur': blank_frames,
            'delay_after': delay_after,
            'duration': dur
        })
    
    # Randomize without consecutive repeats
    trial_list = randomize_without_consecutive_repeats(
        trial_list, 
        key_func=lambda t: (t['uncertainty'], t['interaction'])
    )
    
    # Add trial numbers and block number
    for i, trial in enumerate(trial_list):
        trial['trial_n_in_block'] = i + 1
        trial['block_n'] = 0  # Practice block is block 0
    
    return {
        'block_n': 0,
        'social_condition': 'practice',  # Mixed social and non-social
        'trials': trial_list
    }

"""
TRIGGER HELPER FUNCTIONS
"""
def get_condition_code(trial):
    """
    Generates a 3-digit code identifying the trial condition for LSL triggers.
    Format: [social][uncertainty][interaction]
    
    Social: 1 (social) or 0 (non-social)
    Uncertainty: 5 (high), 2 (low), 1 (certain)
    Interaction: 1 (go) or 0 (no_go)
    
    Examples:
        - social + high + go = 151
        - non_social + certain + no_go = 010
    
    Args:
        trial: Trial dictionary with 'social', 'uncertainty', 'interaction' keys
    
    Returns:
        String with 3-digit code (e.g., "151")
    """
    # Social component
    social_code = '1' if trial['social'] == 'social' else '0'
    
    # Uncertainty component
    uncertainty_map = {
        'high': '5',
        'low': '2',
        'certain': '1'
    }
    uncertainty_code = uncertainty_map.get(trial['uncertainty'], '0')
    
    # Interaction component
    interaction_code = '1' if trial['interaction'] == 'go' else '0'
    
    return social_code + uncertainty_code + interaction_code


"""
MOVEMENT FUNCTIONS
"""
#movement length
def compute_movement_frames(start_pos, end_pos, speed_deg_per_sec = 10.0):
    """Compute the number of frames needed to move from start_pos to end_pos"""
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    distance_deg = math.hypot(dx, dy) #eusclidean distnace in degrees
    duration_sec = distance_deg / speed_deg_per_sec
    n_frames = max(1, int(round(duration_sec / frame_dur))) #min 1 frame
    
    return n_frames

#human-like cursor movement 
def human_movement(win, circles_stims, circles_info, moving_object, partner_stim, hand_stim, outlet, trial, movement_handler=None, condition_code=None):
    """
    Moves the red object to a random green circle using human-like cursor motion.
    Note: This generates movement in pixel space for the trajectory generator,
    but displays in degree space for PsychoPy.
    
    If movement_handler is provided, gives the movement cue at the end of the animation,
    signaling to participants that they should react as fast as possible.
    
    Args:
        partner_stim: The partner stimulus (either social or blurred image)
        hand_stim: The hand stimulus (either social_hand or partner_circle)
        condition_code: 3-digit code for LSL triggers (generated from trial conditions)
    """
    check_escape()
    
    #pick a random green target
    green_circles = [c for c in circles_info if c["color"] == "green"] #list of only green circles
    # Use numpy's random choice for better randomization
    chosen_circle = green_circles[np.random.randint(0, len(green_circles))]
    
    #positions in degrees (PsychoPy's coordinate system)
    start_deg = moving_object_pos
    end_deg = chosen_circle["pos"]
    
    # Convert to PsychoPy internal pixels for trajectory generation
    start_pix_internal = deg2pix(win, start_deg)
    end_pix_internal = deg2pix(win, end_deg)
    
    #simulate human-like cursor movement
    sim_frames = compute_movement_frames(start_deg, end_deg)
    
    # Generate trajectory in internal pixel space
    human_path = HumanizeMouseTrajectory(
        from_point = start_pix_internal, 
        to_point = end_pix_internal, 
        target_points = sim_frames
    )
    trajectory_points_pix_internal = human_path.points
    
    #lsl trigger with condition-specific code
    move_start_time = clock.getTime()  # Record time BEFORE flip
    win.callOnFlip(lambda: outlet.push_sample([f"MOVE_START_{condition_code}"])) #trigger for movement start      
    
        
    #frame locked animation loop
    for i in range(sim_frames):
        check_escape()
        
        # Get position from trajectory (in internal pixels)
        pos_pix_internal = trajectory_points_pix_internal[i]
        
        # Convert internal pixels to degrees for PsychoPy display
        moving_object.pos = pix2deg(win, pos_pix_internal)
    
        #draw initial cues with appropriate hand stimulus
        draw_cues(win, 
                 partner = partner_stim, 
                 social_hand = hand_stim, 
                 moving_object = moving_object, 
                 participant_loc = participant_loc, 
                 action = action, 
                 circles_info = circles_stims)
        win.flip()
    
    move_end_time = clock.getTime()  #record time before flip
    win.callOnFlip(lambda: outlet.push_sample([f"MOVE_END_{condition_code}"])) #trigger for movement end
  
    #ensure final position is exactly in the centre
    moving_object.pos = end_deg
    win.flip()
    
    #calculate actual animation duration
    actual_animation_dur = move_end_time - move_start_time
    expected_animation_dur = sim_frames * frame_dur
    animation_error = (actual_animation_dur - expected_animation_dur) * 1000
    
    if DEBUG_MODE:
        print(f"Animation: expected {expected_animation_dur*1000:.1f}ms, actual {actual_animation_dur*1000:.1f}ms, error {animation_error:+.1f}ms")
    
    #give movement cue to participant
    if movement_handler is not None:
        movement_handler['give_movement_cue']()
    
    #return the position and the index of the target circle
    return chosen_circle["pos"], chosen_circle["index"]


def move_object_back_to_partner(win, circles_stims, moving_object, partner, hand_stim, outlet, trial):
    """
    Moves the object back to the partner location using human-like cursor motion.
    Used in observe trials to show the complete action sequence.
    """
    check_escape()
    
    # Positions in degrees
    start_deg = moving_object.pos
    end_deg = moving_object_pos  # Partner location
    
    # Convert to PsychoPy internal pixels for trajectory generation
    start_pix_internal = deg2pix(win, start_deg)
    end_pix_internal = deg2pix(win, end_deg)
    
    # Simulate human-like cursor movement
    sim_frames = compute_movement_frames(start_deg, end_deg)
    
    # Generate trajectory in internal pixel space
    human_path = HumanizeMouseTrajectory(
        from_point=start_pix_internal,
        to_point=end_pix_internal,
        target_points=sim_frames
    )
    trajectory_points_pix_internal = human_path.points
    
    # LSL trigger for return movement start
    return_start_time = clock.getTime()  # Record time BEFORE flip
    if trial['social'] == 'social':
        win.callOnFlip(lambda: outlet.push_sample(["S_RETURN_START"]))
    else:
        win.callOnFlip(lambda: outlet.push_sample(["NS_RETURN_START"]))
    
    # Frame locked animation loop
    for i in range(sim_frames):
        check_escape()
        
        # Get position from trajectory (in internal pixels)
        pos_pix_internal = trajectory_points_pix_internal[i]
        
        # Convert internal pixels to degrees for PsychoPy display
        moving_object.pos = pix2deg(win, pos_pix_internal)
        
        # Draw all stimuli with appropriate hand stimulus
        draw_cues(win,
             partner=partner,
             social_hand=hand_stim,
             moving_object=moving_object,
             participant_loc=participant_loc,
             action=action,
             circles_info=circles_stims)
        win.flip()
    
    # LSL trigger for return movement end
    return_end_time = clock.getTime()  # Record time BEFORE flip
    if trial['social'] == 'social':
        win.callOnFlip(lambda: outlet.push_sample(["S_RETURN_END"]))
    else:
        win.callOnFlip(lambda: outlet.push_sample(["NS_RETURN_END"]))
    
    # Ensure final position is exact
    moving_object.pos = end_deg
    win.flip()
    
    # Calculate actual return animation duration
    actual_return_dur = return_end_time - return_start_time
    expected_return_dur = sim_frames * frame_dur
    return_error = (actual_return_dur - expected_return_dur) * 1000
    
    if DEBUG_MODE:
        print(f"Return animation: expected {expected_return_dur*1000:.1f}ms, actual {actual_return_dur*1000:.1f}ms, error {return_error:+.1f}ms")
    
    return end_deg
        
#participant movement
def participant_movement(win, mouse, moving_object, participant_loc, outlet, constraint_circle = None, record_trajectory = True, object_proximity_threshold = 0.05, click_anywhere_on_object = True):
    """
    Sets up participant movement tracking with three phases:
    1) Wait in constraint circle, 2) Move to object and click, 3) Drag to participant location.
    Returns a dictionary of functions to control and monitor movement.
    """
    #initialise state variables
    movement_phase = "constrained"
    pick_status = False
    trajectory_recording = False
    trajectory_positions = []
    trajectory_timestamps = []
    
    #reaction time tracking variables
    movement_cue_time = None
    circle_exit_time = None
    reaction_time = None
    cue_given = False
    exited_circle = False
    
    #timing variables
    object_reached_time = None
    drag_start_time = None
    drag_end_time = None
    
    movement_data = {}
    
    def is_mouse_in_circle(circle):
        """Check if mouse is inside the constraint circle"""
        if circle is None:
            return False
        
        circle_radius = circle.radius
        circle_center = circle.pos
        mouse_x, mouse_y = get_corrected_mouse_pos(mouse)
        distance = ((mouse_x - circle_center[0])**2 + (mouse_y - circle_center[1])**2)**0.5
        
        return distance <= circle_radius
    
    def constrain_mouse_to_circle(mouse, circle):
        """Constrain mouse position within a circle boundary"""
        if circle is None:
            return False
            
        circle_radius = circle.radius
        circle_center = circle.pos
        mouse_x, mouse_y = get_corrected_mouse_pos(mouse)
        distance = ((mouse_x - circle_center[0]) ** 2 + (mouse_y - circle_center[1]) ** 2 ) ** 0.5
        
        if distance > circle_radius:
            #calculate constrained position
            angle = math.atan2(mouse_y - circle_center[1], mouse_x - circle_center[0])
            constrained_radius = circle_radius * 0.95 #slightly smaller than the actual circle
            constrained_x = circle_center[0] + constrained_radius * math.cos(angle)
            constrained_y = circle_center[1] + constrained_radius * math.sin(angle)
            
            # Scale up for setPos
            scaled_pos = (constrained_x * SCREEN_SCALE_X, constrained_y * SCREEN_SCALE_Y)
            try:
                mouse.setPos(scaled_pos)
            except AttributeError:
                # Known macOS/pyglet issue - mouse positioning may fail but experiment continues
                pass
            return True
        return False
        
    def check_circle_exit():
        """Detect when participant exits the constraint circle and records reaction time."""
        nonlocal circle_exit_time, reaction_time, exited_circle, movement_phase
        
        if not cue_given or exited_circle or constraint_circle is None:
            return
            
        #check if mouse is outside the circle
        if not is_mouse_in_circle(constraint_circle):
            circle_exit_time = core.getTime()
            reaction_time = circle_exit_time - movement_cue_time
            exited_circle = True
            movement_phase = "moving_to_object"
            
            if DEBUG_MODE:
                print(f"Participant exited constraint circle at: {circle_exit_time}")
                print(f"Reaction time (cue to circle exit): {reaction_time:.3f} seconds")
            
            outlet.push_sample(["CIRCLE_EXIT"]) #reaction time trigger
            
            #start trajectory recording from circle exit
            if record_trajectory:
                nonlocal trajectory_recording, trajectory_positions, trajectory_timestamps
                trajectory_recording = True
                trajectory_positions = [mouse.getPos()]
                trajectory_timestamps = [core.getTime()]
        
    def get_distance_to_object(obj):
        """Get distance from mouse to moving_object."""
        mouse_pos = get_corrected_mouse_pos(mouse)
        obj_pos = obj.pos
        distance = ((mouse_pos[0] - obj_pos[0])**2 + (mouse_pos[1] - obj_pos[1])**2)**0.5 
        return distance
    
    def can_interact_with_object(mouse, obj, click_anywhere_on_object = True, proximity_threshold = moving_object.radius):
        """Check if participant is close enough to the object and clicking to start dragging."""
        mouse_pos = get_corrected_mouse_pos(mouse)
        obj_pos = obj.pos
        distance = ((mouse_pos[0] - obj_pos[0])**2 + (mouse_pos[1] - obj_pos[1])**2)**0.5
        
        mouse1, mouse2, mouse3 = mouse.getPressed()
        
        if click_anywhere_on_object:
            any_button_pressed = mouse1 or mouse2 or mouse3
            close_enough = distance < (obj.radius + proximity_threshold)
            return any_button_pressed and close_enough
        else:
            #requires proximity and click
            return distance < proximity_threshold and (mouse1 or mouse2 or mouse3)
    
    def give_movement_cue():
        """Give the movement cue - participant is allowed to leave constraint circle"""
        nonlocal movement_phase, movement_cue_time, cue_given
        
        movement_cue_time = core.getTime()
        cue_given = True
        movement_phase = "cue_given"
        
        outlet.push_sample(["MOVEMENT_CUE"])
        
        if DEBUG_MODE:
            print(f"Movement cue given at: {movement_cue_time}")
            print("Participant can now leave the constraint circle")
        
    def update_movement():
        """Updates movement state every frame and handles transitions between movement phases."""
        nonlocal movement_phase, pick_status, object_reached_time, drag_start_time, drag_end_time
        nonlocal trajectory_positions, trajectory_timestamps, trajectory_recording
        
        #get mouse state (corrected for Retina scaling)
        mouse_pos = get_corrected_mouse_pos(mouse)
        mouse1, mouse2, mouse3 = mouse.getPressed()
        
        #Phase 1: Constrained to circle
        if movement_phase == "constrained":
            #actively constrain mouse to circle
            if constraint_circle:
                constrain_mouse_to_circle(mouse, constraint_circle)
            return False
            
        #Phase 2: Cue given, waiting for participant to exit circle
        elif movement_phase == "cue_given":
            check_circle_exit()
            
            if not exited_circle and constraint_circle:
                constrain_mouse_to_circle(mouse, constraint_circle)
            
            return False
        
        #Phase 3: Moving to object
        elif movement_phase == "moving_to_object":
            #record trajectory during free movement
            if trajectory_recording and record_trajectory:
                trajectory_positions.append(mouse_pos)
                trajectory_timestamps.append(core.getTime())
                
            #check if participant can interact with the obejct
            if can_interact_with_object(mouse, moving_object, click_anywhere_on_object):
                if object_reached_time is None:
                    #record when they interact for the first time
                    object_reached_time = core.getTime()
                
                movement_phase = "dragging"
                drag_start_time = core.getTime()
                pick_status = True
                
                outlet.push_sample(["RETRIEVAL_START"]) #drag trigger
                
                if DEBUG_MODE:
                    print(f"Dragging started at: {drag_start_time}")
            
            return False
            
        #Phase 4: Dragging object
        elif movement_phase == "dragging":
            if pick_status and (mouse1 or mouse2 or mouse3):
                #update object position while dragging
                moving_object.pos = mouse_pos
            
                #continue recording trajectory
                if trajectory_recording and record_trajectory:
                    trajectory_positions.append(mouse.getPos())
                    trajectory_timestamps.append(core.getTime())
               
                #check for dragging completion
                participant_distance = (
                    (mouse_pos[0] - participant_loc.pos[0])**2 + 
                    (mouse_pos[1] - participant_loc.pos[1])**2)**0.5
                
                if participant_distance < (participant_loc.radius - visual_cursor.radius * 2):
                    drag_end_time = core.getTime()
                    pick_status = False
                    trajectory_recording = False
                    movement_phase = "completed"
            
                    outlet.push_sample(["RETRIEVAL_END"]) #drag end trigger
                    
                    if DEBUG_MODE:
                        print(f"Dragging ended at: {drag_end_time}")
            
                    #calculate trajectory metrics
                    total_trajectory_distance = 0
                    if len(trajectory_positions) > 1:
                        for i in range(1, len(trajectory_positions)):
                            pos1 = trajectory_positions[i-1]
                            pos2 = trajectory_positions[i]
                            total_trajectory_distance += ((pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2)**0.5
            
                    #calculate movement metrics
                    approach_t = object_reached_time - circle_exit_time if object_reached_time and circle_exit_time else None
                    exit_to_drag_time = drag_start_time - circle_exit_time if drag_start_time and circle_exit_time else None
                    total_task_duration = drag_end_time - movement_cue_time if drag_end_time and movement_cue_time else None
                    total_movement_duration = drag_end_time - circle_exit_time if drag_end_time and circle_exit_time else None
                    drag_duration = drag_end_time - drag_start_time if drag_start_time else None
                    
                    #prepare movement data
                    movement_data.update({
                        #movement cue and rt
                        'movement_cue_time': movement_cue_time,
                        'circle_exit_time': circle_exit_time,
                        'rt': reaction_time,
                        'cue_given': cue_given,
                        'exited_circle': exited_circle,
                        
                        #task timing
                        'object_reached_time': object_reached_time,
                        'drag_start_time': drag_start_time,
                        'drag_end_time': drag_end_time,
                        
                        #duration calculations
                        'exit_to_drag_time': exit_to_drag_time,
                        'total_task_duration':total_task_duration,
                        'mt': total_movement_duration,
                        'approach_t': approach_t,
                        'drag_t': drag_duration,
                        
                        #trajectory data
                        'trajectory_positions': trajectory_positions,
                        'trajectory_timestamps': trajectory_timestamps,
                        'total_trajectory_distance': total_trajectory_distance,
                        
                        #status
                        'completed': True,
                        'phase': movement_phase,
                        'success': True
                        })
            
                    return True  # Movement completed
                else:
                    #mouse button released before target location
                    if not (mouse1 or mouse2 or mouse3) and DEBUG_MODE:
                        print("Mouse button released during drag")
        
            return False  # Movement still in progress
        
        #Phase 5: Completed
        elif movement_phase == "completed":
            return True
        return False
        
    def reset_movement():
        """Reset all movement state variables"""
        nonlocal movement_phase, pick_status, trajectory_recording, trajectory_positions
        nonlocal trajectory_timestamps, movement_cue_time, circle_exit_time, reaction_time
        nonlocal cue_given, exited_circle, object_reached_time, drag_start_time, drag_end_time
        
        movement_phase = "constrained"
        pick_status = False
        trajectory_recording = False
        trajectory_positions = []
        trajectory_timestamps = []
        movement_cue_time = None
        circle_exit_time = None
        reaction_time = None
        cue_given = False
        exited_circle = False
        object_reached_time = None
        drag_start_time = None
        drag_end_time = None
    
    
    #return the update function and current state
    return {
        'update': update_movement,
        'give_movement_cue': give_movement_cue,
        'is_dragging': lambda: pick_status,
        'get_phase': lambda: movement_phase,
        'get_data': lambda: movement_data,
        'get_distance_to_object': lambda: get_distance_to_object(moving_object),
        'get_reaction_time': lambda: reaction_time,
        'has_exited_circle': lambda: exited_circle,
        'cue_was_given': lambda: cue_given,
        'is_constrained': lambda: movement_phase in ["constrained", "cue_given"] and not exited_circle,
        'reset': reset_movement
        }
    
"""
TRIAL EXECUTION
"""
def initialize_mouse_position(mouse, target_circle):
    """
    Initialize mouse position to the center of the target circle.
    Accounting for Retina scaling when setting the position.
    """
    if target_circle:
        center_pos = target_circle.pos
        
        #scale up the position for setPos (opposite of getPos correction)
        scaled_pos = (center_pos[0] * SCREEN_SCALE_X, center_pos[1] * SCREEN_SCALE_Y)
        
        try:
            mouse.setPos(scaled_pos)
            if DEBUG_MODE:
                print(f"Mouse initialized to position (degrees): {center_pos}")
                print(f"Scaled position sent to mouse.setPos: {scaled_pos}")
            
            #verify the actual position
            actual_pos = get_corrected_mouse_pos(mouse)
            if DEBUG_MODE:
                print(f"Corrected mouse position after setPos: {actual_pos}")
        except (AttributeError, TypeError) as e:
            if DEBUG_MODE:
                print(f"⚠️  Warning: Could not programmatically set mouse position (macOS/Pyglet bug)")
                print(f"Mouse will start from current position instead")
        
        return True
    return False

def run_trial(trial, outlet, clock):
    """
    Runs 1 trial with the following sequence:
    1. Fixation cross, 2. Blank screen, 3. Cue presentation, 
    4. Animated movement, 5. Participant response (if retrieve trial)
    Sends LSL triggers and collects behavioral data.
    """
    
    #generate condition code
    condition_code = get_condition_code(trial)
    
    #clear previous keyboard inputs
    event.clearEvents(eventType='keyboard')
    
    #set correct partner stim based on trial
    if trial['social'] == "social":
        partner_img.image = img_partner_social
        partner = partner_img
        hand_stim = social_hand 
    else:
        partner_img.image = img_partner_non_social 
        partner = partner_img 
        hand_stim = partner_circle 
    
    #select number of green circles based on the condition
    if trial['uncertainty'] == "certain":
        num_green = 1
    elif trial['uncertainty'] == "low":
        num_green = 2
    else:
        num_green = 5
    
    #select correct response action image
    action.image = img_go if trial['interaction'] == "go" else img_nogo
    
    #reset moving_object position
    moving_object.pos = moving_object_pos
        
    #prepare circle stimuli
    circles_info, green_circle_positions = draw_location_circles(win, num_circles=5, radius=0.35, spacing=3, y_pos=0, n_green=num_green)
    
    circles_stims = []
    for circle in circles_info:
        stim = visual.Circle(win, radius=0.5, pos=circle["pos"], fillColor=circle["color"], lineColor=circle["color"])
        circles_stims.append(stim)
    
    #fixation (baseline)
    fix_cross.draw()
    
    #align markers to the win.flip
    #record time before the flip to match when trigger is sent
    fix_start_time = clock.getTime()
    win.callOnFlip(lambda: outlet.push_sample([f"TRIAL_START_{condition_code}"]))
    win.callOnFlip(lambda: outlet.push_sample([f"FIX_CROSS_{condition_code}"])) #fixation trigger
    win.flip()
    
    #frame-based wait for fixation
    safe_wait(trial['fix_dur'], fix_cross = fix_cross)
    check_escape() #check for escape key
    
    #blank screen
    blank_start_time = clock.getTime()
    win.callOnFlip(lambda: outlet.push_sample([f"BLANK_SCREEN_PRE_CUE_{condition_code}"]))
    win.flip()
    
    #measure fixation duration: time from fix trigger to blank trigger
    actual_fix_dur = blank_start_time - fix_start_time

    #time-based wait for blank screen
    time_based_wait(frames_to_secs(trial['blank_dur']))
    check_escape()

    # Draw cues with appropriate hand stimulus based on condition
    draw_cues(win, partner = partner, social_hand = hand_stim, moving_object = moving_object, participant_loc = participant_loc, action = action, circles_info = circles_stims)
    # Record time BEFORE flip to match when trigger is sent
    cue_start_time = clock.getTime()
    win.callOnFlip(lambda: outlet.push_sample([f"CUE_SHOW_{condition_code}"])) #trigger for cue presentation start
    win.flip() #showing all drawn stim together
    
    # NOW measure blank duration: time from blank trigger to cue trigger (matches LSL)
    actual_blank_dur = cue_start_time - blank_start_time
    
    # Frame-based wait for cue (need to redraw stimuli)
    safe_wait(trial['duration'], partner = partner, social_hand = hand_stim, moving_object = moving_object, participant_loc = participant_loc, action = action, circles_info = circles_stims)
    cue_end_time = clock.getTime()
    actual_cue_dur = cue_end_time - cue_start_time
    
    check_escape()
    
    #initialise trial results
    trial_results = {
        #basic movement metrics
        'rt': None,
        'approach_t': None,
        'drag_t': None,
        'mt': None,
        'success': False,
        
        #movement cue and rt
        'movement_cue_time': None,
        'circle_exit_time': None,
        
        #duration calculations
        'exit_to_drag_time': None,
        'total_task_duration': None,
                        
        #trajectory data
        'trajectory_positions': None,
        'trajectory_timestamps': None,
        'total_trajectory_distance': None,
        
        #task timing
        'object_reached_time': None,
        'drag_start_time': None,
        'drag_end_time': None,
        'movement_completed': False,
        'timeout_occurred': False,
        
        #constraint and phase data
        'cue_given': False,
        'exited_circle': False,
        'final_movement_phase': None,
        'was_constrained': False
    }
    
    #movement phase - setup movement handler BEFORE animation
    movement_handler = None
    if trial['task'] == 'retrieve':
        #keep system cursor hidden - using custom white circle cursor instead
        mouse.setVisible(False)
        
        #initialise mouse to participant loc
        initialize_mouse_position(mouse, participant_loc)
        
        movement_handler = participant_movement(
            win = win, 
            mouse = mouse, 
            moving_object = moving_object, 
            participant_loc = participant_loc, 
            outlet = outlet, 
            constraint_circle = participant_loc, 
            record_trajectory = True, 
            click_anywhere_on_object = True)
    
    #animation of movement - movement cue will be given at the END of this animation
    final_pos, target_circle_index = human_movement(win, circles_stims, circles_info, moving_object, partner, hand_stim, outlet, trial, movement_handler, condition_code)
    
    #movement loop (only for retrieve trials)
    if trial['task'] == 'retrieve':
        movement_completed = False
        max_time = 10.0
        start_time = clock.getTime()
        
        # Initialize mouse tracking data collection
        mouse_tracking_data = {
            'timestamps': [],
            'positions_deg': [],
            'positions_corrected_deg': [],
            'buttons_pressed': []
        }
        
        # Flag to start recording only after circle exit
        recording_started = False
        
        while not movement_completed and (clock.getTime() - start_time) < max_time:
            check_escape()
            
            #update movement state
            movement_completed = movement_handler['update']()
            
            # Start recording only after participant exits the constraint circle
            if not recording_started and movement_handler['has_exited_circle']():
                recording_started = True
                if DEBUG_MODE:
                    print("Started recording mouse tracking data")
            
            # Record mouse tracking data every frame (only after movement starts)
            if recording_started:
                current_time = clock.getTime()
                raw_pos = mouse.getPos()
                corrected_pos = get_corrected_mouse_pos(mouse)
                buttons = mouse.getPressed()
                
                mouse_tracking_data['timestamps'].append(current_time)
                mouse_tracking_data['positions_deg'].append(corrected_pos)
                mouse_tracking_data['positions_corrected_deg'].append(corrected_pos)
                mouse_tracking_data['buttons_pressed'].append(buttons)
            
            event.clearEvents() #clear events
            
            #draw stimuli with appropriate hand stimulus
            draw_cues(win, 
                partner = partner, 
                social_hand = hand_stim, 
                moving_object = moving_object, 
                participant_loc = participant_loc, 
                action = action, 
                circles_info = circles_stims,
                mouse = mouse,
                show_cursor = True)
            
            win.flip()
        
        #hide system cursor after movement is complete or timed out
        mouse.setVisible(False)
        
        if movement_completed:
            movement_data = movement_handler['get_data']()
            trial_results.update({
                #basic movement metrics
                'rt': movement_data.get('rt'),
                'approach_t': movement_data.get('approach_t'),
                'drag_t': movement_data.get('drag_t'),
                'mt': movement_data.get('mt'),
                'success': movement_data.get('success', False),
                
                #circle exit rt measures
                'movement_cue_time': movement_data.get('movement_cue_time'),
                'circle_exit_time': movement_data.get('circle_exit_time'),
                
                #movement phase timing
                'exit_to_drag_time': movement_data.get('exit_to_drag_time'),
                'total_task_duration': movement_data.get('total_task_duration'),
                
                #trajectory data
                'trajectory_positions': movement_data.get('trajectory_positions'),
                'trajectory_timestamps': movement_data.get('trajectory_timestamps'),
                'total_trajectory_distance': movement_data.get('total_trajectory_distance'),
                
                #task timing
                'object_reached_time': movement_data.get('object_reached_time'),
                'drag_start_time': movement_data.get('drag_start_time'),
                'drag_end_time': movement_data.get('drag_end_time'),
                'movement_completed': True,
                'timeout_occurred': False,
                
                #constraint and phase data
                'cue_given': movement_data.get('cue_given', False),
                'exited_circle': movement_data.get('exited_circle', False),
                'final_movement_phase': movement_data.get('phase'),
                'was_constrained': True,
                
                #detailed mouse tracking data (frame-by-frame)
                'mouse_tracking_timestamps': mouse_tracking_data['timestamps'],
                'mouse_tracking_positions': mouse_tracking_data['positions_deg'],
                'mouse_tracking_buttons': mouse_tracking_data['buttons_pressed']
                })
        else:
            #handle timeout or incomplete movement
            partial_data = movement_handler['get_data']()
            trial_results.update({
                'movement_completed': False,
                'timeout_occurred': True,
                'final_movement_phase': movement_handler['get_phase'](),
                'cue_given': movement_handler['cue_was_given'](),
                'exited_circle': movement_handler['has_exited_circle'](),
                'reaction_time_circle_exit': movement_handler['get_reaction_time'](),
                'was_constrained': movement_handler['is_constrained'](),
                
                #detailed mouse tracking data (even for incomplete trials)
                'mouse_tracking_timestamps': mouse_tracking_data['timestamps'],
                'mouse_tracking_positions': mouse_tracking_data['positions_deg'],
                'mouse_tracking_buttons': mouse_tracking_data['buttons_pressed']
            })
            if DEBUG_MODE:
                print("Movement timed out or was incomplete")
    #observe condition - object moves back to partner
    else:
        # Track initial mouse position to detect any movement (no-go condition)
        initial_mouse_pos = mouse.getPos()
        mouse_moved = False
        movement_threshold = 0.5  # degrees - minimum movement to count as "moved"
        
        #short wait to allow participants to observe final position
        delay_start_time = clock.getTime()
        
        # Monitor for mouse movement during the delay period
        start_wait_time = clock.getTime()
        target_wait_time = start_wait_time + frames_to_secs(trial['delay_after'])
        
        while clock.getTime() < target_wait_time:
            check_escape()
            
            # Check if mouse has moved significantly
            current_mouse_pos = mouse.getPos()
            distance_moved = ((current_mouse_pos[0] - initial_mouse_pos[0])**2 + 
                            (current_mouse_pos[1] - initial_mouse_pos[1])**2)**0.5
            
            if distance_moved > movement_threshold and not mouse_moved:
                mouse_moved = True
                if DEBUG_MODE:
                    print(f"⚠️  NO-GO FAIL: Participant moved mouse by {distance_moved:.2f} degrees")
            
            # Draw stimuli with appropriate hand stimulus
            draw_cues(win, partner = partner, social_hand = hand_stim, moving_object = moving_object, 
                     participant_loc = participant_loc, action = action, circles_info = circles_stims)
            
            win.flip()
        
        delay_end_time = clock.getTime()
        actual_delay_after = delay_end_time - delay_start_time
        
        #move object back to partner using human-like movement
        final_pos = move_object_back_to_partner(win, circles_stims, moving_object, partner, hand_stim, outlet, trial)
        
        #no-go trial results (participant should not interact)
        trial_results.update({
            'movement_completed': False,
            'timeout_occurred': False,
            'final_movement_phase': 'observe',
            'cue_given': False,
            'exited_circle': False,
            'was_constrained': False,
            'success': not mouse_moved,  # Success if they didn't move (no-go)
            'no_go_success': not mouse_moved  # Keep for backward compatibility in unified success logic
        })

    final_pos = moving_object.pos #final position after movement
    win.flip()
    
    #check for escape key
    check_escape()
    
    #trial end marker
    trial_end_time = clock.getTime()
    outlet.push_sample([f"TRIAL_END_{condition_code}"])
    
    # ===== COMPREHENSIVE TIMING SUMMARY FOR ALL TRIGGERS =====
    # Only print in DEBUG mode - printing is slow and affects timing performance
    if DEBUG_MODE:
        print(f"\n{'='*60}")
        print(f"TRIAL {trial.get('trial_number', '?')} - ALL TRIGGER TIMINGS")
        print(f"{'='*60}")
        
        # Core trial structure timings
        print(f"\n--- Core Trial Structure ---")
        print(f"FIX_CROSS:            {actual_fix_dur*1000:.1f}ms (expected: {trial['fix_dur']*frame_dur*1000:.1f}ms, error: {(actual_fix_dur - trial['fix_dur']*frame_dur)*1000:+.1f}ms)")
        print(f"BLANK_SCREEN_PRE_CUE: {actual_blank_dur*1000:.1f}ms (expected: {trial['blank_dur']*frame_dur*1000:.1f}ms, error: {(actual_blank_dur - trial['blank_dur']*frame_dur)*1000:+.1f}ms)")
        print(f"CUE_SHOW:             {actual_cue_dur*1000:.1f}ms (expected: {trial['duration']*frame_dur*1000:.1f}ms, error: {(actual_cue_dur - trial['duration']*frame_dur)*1000:+.1f}ms)")
        
        # Observe trial specific timing
        if trial['task'] == 'observe':
            print(f"\n--- Observe Trial Timing ---")
            print(f"DELAY_AFTER:          {actual_delay_after*1000:.1f}ms")
        
        # Retrieve trial specific timing
        if trial['task'] == 'retrieve' and trial_results['movement_completed']:
            print(f"\n--- Retrieve Trial Timing ---")
            if trial_results.get('movement_cue_time'):
                print(f"MOVEMENT_CUE:         at {trial_results['movement_cue_time']:.3f}s")
            if trial_results.get('circle_exit_time'):
                print(f"CIRCLE_EXIT:          at {trial_results['circle_exit_time']:.3f}s")
                if trial_results.get('rt'):
                    print(f"  → Reaction time:    {trial_results['rt']*1000:.1f}ms")
            if trial_results.get('drag_start_time'):
                print(f"RETRIEVAL_START:      at {trial_results['drag_start_time']:.3f}s")
            if trial_results.get('drag_end_time'):
                print(f"RETRIEVAL_END:        at {trial_results['drag_end_time']:.3f}s")
            if trial_results.get('mt'):
                print(f"  → Total movement time: {trial_results['mt']*1000:.1f}ms")
        
        # Trial boundaries
        print(f"\n--- Trial Boundaries ---")
        print(f"TRIAL_START:          at {fix_start_time:.3f}s")
        print(f"TRIAL_END:            at {trial_end_time:.3f}s")
        print(f"Total trial duration: {(trial_end_time - fix_start_time)*1000:.1f}ms")
        print(f"{'='*60}\n")
    
    #record trial outcome
    trial_data = trial.copy()
    trial_data.update({
        'timestamp': clock.getTime(), 
        'final_position': final_pos,
        'participant_id': subj_info['ID'],
        'age': subj_info['age'],
        'gender': subj_info['gender'],
        'native_language': subj_info['native language'],
        'block_n': trial.get('block_n'),
        'handedness': subj_info['handedness'],
        
        #actual measured timing in seconds (from clock, not frame conversion)
        'fix_dur': actual_fix_dur,
        'blank_dur': actual_blank_dur,
        'delay_after': actual_delay_after if trial['task'] == 'observe' else None,
        'duration': actual_cue_dur,
        
        # PsychoPy clock timestamps for each trigger (for EEG synchronization validation)
        'fix_start_time': fix_start_time,
        'blank_start_time': blank_start_time,
        'cue_start_time': cue_start_time,
        'trial_end_time': trial_end_time,
        'cue_end_time': cue_end_time,
        
        'rt': trial_results['rt'],
        'approach_t': trial_results['approach_t'],
        'drag_t': trial_results['drag_t'],
        'mt': trial_results['mt'],
        # Unified success column: True if retrieve completed OR if no-go succeeded (didn't move)
        'success': trial_results.get('no_go_success', trial_results['success']),
        'movement_cue_time': trial_results['movement_cue_time'],
        'circle_exit_time': trial_results['circle_exit_time'],
        'exit_to_drag_time': trial_results['exit_to_drag_time'],
        'total_task_duration': trial_results['total_task_duration'],
        'trajectory_positions': trial_results['trajectory_positions'],
        'trajectory_timestamps': trial_results['trajectory_timestamps'],
        'total_trajectory_distance': trial_results['total_trajectory_distance'],
        'object_reached_time': trial_results['object_reached_time'],
        'drag_start_time': trial_results['drag_start_time'],
        'drag_end_time': trial_results['drag_end_time'],
        'movement_completed': trial_results['movement_completed'],
        'timeout_occurred': trial_results['timeout_occurred'],
        'cue_given': trial_results['cue_given'],
        'exited_circle': trial_results['exited_circle'],
        'final_movement_phase': trial_results['final_movement_phase'],
        'was_constrained': trial_results['was_constrained'],
        
        # Circle position information (numbered 0-4 from left to right)
        'green_circle_positions': green_circle_positions,  # List of indices of green circles
        'target_circle_index': target_circle_index,  # Index (1-5) of the actual target circle chosen
        
        #detailed mouse tracking data
        'mouse_tracking_timestamps': trial_results.get('mouse_tracking_timestamps'),
        'mouse_tracking_positions': trial_results.get('mouse_tracking_positions'),
        'mouse_tracking_buttons': trial_results.get('mouse_tracking_buttons')
        })
    
    return trial_data
    
"""
MAIN EXPERIMENT LOOP
"""
#intro and outro text
text_pos = [0, 0] #position of the text
text_height = 0.3 # height in degrees

# Split intro text into multiple screens for better readability
intro_text = [
    # Screen 1: Welcome and Overview
    u"""Introduction and experimental task:

Welcome to my BSc Project study on uncertainty in motor planning and coordination in joint action.

During the experiment you will observe movement of an objects on a computer screen and try to coordinate your actions accordingly. 


Press SPACE to continue...""",

    # Screen 2: Experimental Scenarios
    u"""TWO EXPERIMENTAL SCENARIOS:

Throughout the experiment, you will participate in blocks of trials that fall into two distinct categories. Your core task remains the same, but the source diving the object movement will change.

Joint Action Scenario (Social Condition)
In this scenario your task is to engage in joint action with a partner. The movement of a red object is controlled and guided by this partner to one of the available target locations (green circles).  

Independent Action Scenario (Non-social Condition)
In this scenario, the red object moves independently, guided by a pre-programmed non-intentional algorithm. This algorithm determines the object's path and final target location randomly within the available green circles.

Social partner condition will be indicated with a picture of a face in the top middle portion of the screen and a hand symbol to the left from the image. Non-social condition will be represented with a blurred image in the same location as face image and a grey circle in place of a social hand.


Press SPACE to continue...""",

    # Screen 3: Types of Actions
    u"""TYPES OF ACTIONS

In both scenarios, movement of the red object follows one of the two sequences:

Individual Action
The individual action involves the red object being moved to an available target location (green circle). Subsequently, the object is collected and moved back to a starting position (upper centre of the screen). Your response is not required.

Individual actions are indicated by a hand with a cross.

Interactive Action
The interactive action involves the partner or the algorithm moving the red object from a starting position to an available target location. Subsequently, you must collect the red object my moving the mouse towards it, click on it and move it to your character location (grey circle at the lower centre of the screen).

Keep in mind that the number of green targets will vary between trials.

Interactive rounds are indicated by a single hand symbol to the right from the face or blurred image.


Press SPACE to continue...""",

    # Screen 4: Your Task
    u"""YOUR TASK:

It's important that you try to predict the actions of your partner or the movement of the red object to optimise your own actions.

Try to respond as fast and accurately as possible through retrieving the object to your own location. 

Retrieval of the red object can be only started after the movement of the object has been finished and its location is within one of the green circles.


Press SPACE when you're ready to proceed to a trial block of the experiment. You will be notified once the actual experiment starts."""
]

break_text = [u"Block completed!\n\nTake a break if you need.\n\nPress SPACE when ready to continue."]
outro_text = [u"Thank you for participating in the study! :)"]

def save_block_data(block_results, block_n, participant_id, current_date):
    """
    Save data for a single block to CSV files.
    
    Args:
        block_results: List of trial data dictionaries for this block
        block_n: Block number
        participant_id: Participant ID
        current_date: Current date string (YYYY-MM-DD format)
    """
    #create directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    #save main trial data
    df = pd.DataFrame(block_results)
    filename = os.path.join(DATA_DIR, f"participant_{participant_id}_block{block_n}_{current_date}_data.csv")
    df.to_csv(filename, index=False)
    if DEBUG_MODE:
        print(f"Block {block_n} data saved to {filename}")
    
    #save detailed mouse tracking data to separate file
    mouse_tracking_rows = []
    for trial_data in block_results:
        if trial_data.get('mouse_tracking_timestamps'):
            timestamps = trial_data['mouse_tracking_timestamps']
            positions = trial_data['mouse_tracking_positions']
            buttons = trial_data['mouse_tracking_buttons']
            
            for i in range(len(timestamps)):
                row = {
                    'participant_id': participant_id,
                    'block_n': block_n,
                    'trial_n_in_block': trial_data.get('trial_n_in_block', trial_data.get('trial_n')),
                    'social': trial_data['social'],
                    'uncertainty': trial_data['uncertainty'],
                    'interaction': trial_data['interaction'],
                    'task': trial_data['task'],
                    'timestamp': timestamps[i],
                    'mouse_x_deg': positions[i][0],
                    'mouse_y_deg': positions[i][1],
                    'button_left': buttons[i][0],
                    'button_middle': buttons[i][1],
                    'button_right': buttons[i][2],
                    'frame_number': i
                }
                mouse_tracking_rows.append(row)
    
    if mouse_tracking_rows:
        df_tracking = pd.DataFrame(mouse_tracking_rows)
        tracking_filename = os.path.join(DATA_DIR, f"participant_{participant_id}_block{block_n}_{current_date}_mouse_tracking.csv")
        df_tracking.to_csv(tracking_filename, index=False)
        if DEBUG_MODE:
            print(f"Block {block_n} mouse tracking data saved to {tracking_filename}")
            print(f"Total tracking samples in block {block_n}: {len(mouse_tracking_rows)}")


def append_block_data(block_results, participant_id, current_date):
    """
    Append block data to existing CSV files (or create them if they don't exist).
    This ensures data is saved incrementally and not lost if experiment terminates early.
    
    Args:
        block_results: List of trial data dictionaries for this block
        participant_id: Participant ID
        current_date: Current date string
    """
    #create directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    filename = os.path.join(DATA_DIR, f"participant_{participant_id}_{current_date}_data.csv")
    tracking_filename = os.path.join(DATA_DIR, f"participant_{participant_id}_{current_date}_mouse_tracking.csv")
    
    #save main trial data (append if file exists, create with header if not)
    df = pd.DataFrame(block_results)
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)
    
    if DEBUG_MODE:
        print(f"Block data appended to {filename}")
    
    #save detailed mouse tracking data
    mouse_tracking_rows = []
    for trial_data in block_results:
        if trial_data.get('mouse_tracking_timestamps'):
            timestamps = trial_data['mouse_tracking_timestamps']
            positions = trial_data['mouse_tracking_positions']
            buttons = trial_data['mouse_tracking_buttons']
            
            for i in range(len(timestamps)):
                row = {
                    'participant_id': participant_id,
                    'block_n': trial_data.get('block_n'),
                    'trial_n_in_block': trial_data.get('trial_n_in_block', trial_data.get('trial_n')),
                    'social': trial_data['social'],
                    'uncertainty': trial_data['uncertainty'],
                    'interaction': trial_data['interaction'],
                    'task': trial_data['task'],
                    'timestamp': timestamps[i],
                    'mouse_x_deg': positions[i][0],
                    'mouse_y_deg': positions[i][1],
                    'button_left': buttons[i][0],
                    'button_middle': buttons[i][1],
                    'button_right': buttons[i][2],
                    'frame_number': i
                }
                mouse_tracking_rows.append(row)
    
    if mouse_tracking_rows:
        df_tracking = pd.DataFrame(mouse_tracking_rows)
        if os.path.exists(tracking_filename):
            df_tracking.to_csv(tracking_filename, mode='a', header=False, index=False)
        else:
            df_tracking.to_csv(tracking_filename, mode='w', header=True, index=False)
        
        if DEBUG_MODE:
            print(f"Mouse tracking data appended to {tracking_filename}")

def run_experiment():
    """
    Complete experiment flow with block design:
        - intro text
        - loop through blocks:
            - run trials in block
            - show break screen (except after last block)
            - save data after every block
        - outro
    """
    #get current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    win.flip()
    
    #intro text - display multiple screens
    for screen_idx, text in enumerate(intro_text):
        intro = visual.TextStim(
            win, 
            text=text, 
            pos=text_pos, 
            height=text_height,
            wrapWidth=13,  #wrap text to fit on screen
            alignText='left'  
        )
        intro.draw()
        win.flip()
        event.waitKeys(keyList=['space'])
        check_escape()
    
    #create experiment blocks
    practice_block = make_practice_block()
    blocks = make_experiment_blocks(num_blocks=8, trials_per_block=30)
    
    #combine practice block with experimental blocks
    all_blocks = [practice_block] + blocks
    
    if DEBUG_MODE:
        print(f"\n=== EXPERIMENT STRUCTURE ===")
        print(f"Practice block: {practice_block['social_condition']} ({len(practice_block['trials'])} trials - all 12 conditions)")
        print(f"Experimental blocks: {len(blocks)}")
        for block in blocks:
            print(f"  Block {block['block_n']}: {block['social_condition']} ({len(block['trials'])} trials)")
        print("="*40 + "\n")
    
    #accumulate all results across blocks
    all_results = []
    
    #loop through all blocks (practice + experimental)
    for block_idx, block in enumerate(all_blocks):
        block_n = block['block_n']
        social_condition = block['social_condition']
        trials = block['trials']
        
        #show practice block intro message
        if block_n == 0:
            practice_intro = visual.TextStim(
                win,
                text="PRACTICE BLOCK\n\nYou will now complete 12 practice trials covering all experimental conditions.\n\nThis will help you familiarize yourself with the task.\n\nPress SPACE to begin practice.",
                pos=text_pos,
                height=text_height,
                wrapWidth=25
            )
            practice_intro.draw()
            win.flip()
            event.waitKeys(keyList=['space'])
            check_escape()
        
        if DEBUG_MODE:
            if block_n == 0:
                print(f"\n--- Starting PRACTICE Block (all 12 conditions) ---")
            else:
                print(f"\n--- Starting Block {block_n} ({social_condition}) ---")
        
        #send block start marker
        block_start_time = clock.getTime()
        outlet.push_sample([f"BLOCK_{block_n}_START_{social_condition.upper()}"])
        if DEBUG_MODE:
            print(f"BLOCK_{block_n}_START trigger sent at: {block_start_time:.3f}s")
        
        #loop through trials in this block
        for trial in trials:
            data_out = run_trial(trial, outlet, clock)
            data_out.update(subj_info)
            all_results.append(data_out) #add to combined list
            
            #check for escape key to quit
            check_escape()
        
        #send block end marker
        block_end_time = clock.getTime()
        outlet.push_sample([f"BLOCK_{block_n}_END"])
        block_duration = block_end_time - block_start_time
        if DEBUG_MODE:
            print(f"BLOCK_{block_n}_END trigger sent at: {block_end_time:.3f}s")
            print(f"Block {block_n} duration: {block_duration/60:.1f} minutes ({block_duration:.1f}s)")
            print(f"--- Block {block_n} completed ({len(trials)} trials) ---\n")
        
        #append block data to main CSV files after every block
        block_results = [trial for trial in all_results if trial.get('block_n') == block_n]
        append_block_data(block_results, subj_info['ID'], current_date)
        
        #show break screen between blocks 
        if block_idx < len(all_blocks) - 1:
            #message after practice block
            if block_n == 0:
                break_msg = "Practice block completed!\n\nThe actual experiment will now begin.\n\nPress SPACE when ready to continue."
            else:
                break_msg = break_text[0]
            
            break_screen = visual.TextStim(
                win, 
                text=break_msg, 
                pos=text_pos, 
                height=text_height
            )
            break_screen.draw()
            win.flip()
            event.waitKeys(keyList=['space'])
            
            #check for escape during break
            check_escape()
    
    if DEBUG_MODE:
        print("\n=== ALL BLOCKS COMPLETED ===")
        print(f"All data already saved to: {DATA_DIR}")
    
    #outro text
    outro = visual.TextStim(win, text=outro_text[0], pos=text_pos, height=text_height)
    outro.draw()
    win.flip()
    core.wait(2)
    
    #check for escape key to quit
    check_escape()
    
    if DEBUG_MODE:
        print("\n=== EXPERIMENT COMPLETED ===")
        print(f"All data saved to: {DATA_DIR}")
        print(f"Participant: {subj_info['ID']}")
        print(f"Date: {current_date}")
        print(f"Total trials: {len(all_results)}")
        print(f"Total blocks: {len(blocks)}")
        print(f"Files created:")
        print(f"  - participant_{subj_info['ID']}_{current_date}_data.csv")
        print(f"  - participant_{subj_info['ID']}_{current_date}_mouse_tracking.csv")
    
    #close everything 
    win.close()
    core.quit()
    
"""
MAIN
"""
if __name__ == "__main__": 
    run_experiment()
    