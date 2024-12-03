import sys
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import mplcursors
from scipy.stats import chi2
from scipy.optimize import linear_sum_assignment
from PyQt5.QtWidgets import (QApplication, QWidget, QTableWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QComboBox, QTextEdit,
                             QHBoxLayout, QSplitter, QCheckBox, QLineEdit, QDialog, QGridLayout, QGroupBox, QRadioButton,
                             QFrame, QSizePolicy, QToolButton, QTabWidget, QMenu, QAction, QTableWidgetItem, QScrollArea)
from PyQt5.QtGui import QFont, QIcon, QColor
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QObject

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import pyqtgraph as pg

# Custom stream class to redirect stdout
class OutputStream:
    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, text):
        self.text_edit.append(text)

    def flush(self): 
        pass  # No need to implement flush for QTextEdit

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))  # Predicted state vector
        self.Pp = np.eye(6)  # Predicted state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.prev_Time = 0
        self.Q = np.eye(6)
        self.Phi = np.eye(6)
        self.Z = np.zeros((3, 1))
        self.Z1 = np.zeros((3, 1))  # Measurement vector
        self.Z2 = np.zeros((3, 1))
        self.first_rep_flag = False
        self.second_rep_flag = False
        self.gate_threshold = 900.21  # 95% confidence interval for Chi-squared distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        print(f"Initializing filter state with x: {x}, y: {y}, z: {z}, vx: {vx}, vy: {vy}, vz: {vz}, time: {time}")
        if not self.first_rep_flag:
            self.Z1 = np.array([[x], [y], [z]])
            self.Sf[0] = x
            self.Sf[1] = y
            self.Sf[2] = z
            print("check sfffffffffffffff",self.Sf[0])
            self.Meas_Time = time
            self.prev_Time = self.Meas_Time
            self.first_rep_flag = True
        elif self.first_rep_flag and not self.second_rep_flag:
            self.Z2 = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time
            dt = self.Meas_Time - self.prev_Time
            self.Sf[3] = (self.Z2[0] - self.Z1[0]) / dt
            self.Sf[4] = (self.Z2[1] - self.Z1[1]) / dt
            self.Sf[5] = (self.Z2[2] - self.Z1[2]) / dt
            self.second_rep_flag = True
        else:
            self.Z = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time

    def predict_step(self, current_time):
        dt = current_time - self.prev_Time
        print(f"Predict step with dt: {dt}")
        T_2 = (dt * dt) / 2.0
        T_3 = (dt * dt * dt) / 3.0
        self.Phi[0, 3] = dt
        self.Phi[1, 4] = dt
        self.Phi[2, 5] = dt
        self.Q[0, 0] = T_3
        self.Q[1, 1] = T_3
        self.Q[2, 2] = T_3
        self.Q[0, 3] = T_2
        self.Q[1, 4] = T_2
        self.Q[2, 5] = T_2
        self.Q[3, 0] = T_2
        self.Q[4, 1] = T_2
        self.Q[5, 2] = T_2
        self.Q[3, 3] = dt
        self.Q[4, 4] = dt
        self.Q[5, 5] = dt
        self.Q = self.Q * self.plant_noise
        self.Sp = np.dot(self.Phi, self.Sf)
        self.Pp = np.dot(np.dot(self.Phi, self.Pf), self.Phi.T) + self.Q
        self.Meas_Time = current_time

    def update_step(self, Z):
        print(f"Update step with measurement Z: {Z}")
        Inn = Z - np.dot(self.H, self.Sp)
        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pp, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sp + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pp)

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[10])  # MR column
            ma = float(row[11])  # MA column
            me = float(row[12])  # ME column
            mt = float(row[13])  # MT column
            md = float(row[14])
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            print(f"Converted spherical to Cartesian: azimuth={ma}, elevation={me}, range={mr} -> x={x}, y={y}, z={z}")
            measurements.append((mr, ma, me, mt, md, x, y, z))
    return measurements

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
    az = math.atan2(y, x)

    if x > 0.0:
        az = np.pi / 2 - az
    else:
        az = 3 * np.pi / 2 - az

    az = az * 180 / np.pi

    if az < 0.0:
        az = 360 + az

    if az > 360:
        az = az - 360

    print(f"Converted Cartesian to spherical: x={x}, y={y}, z={z} -> range={r}, azimuth={az}, elevation={el}")
    return r, az, el

def form_measurement_groups(measurements, max_time_diff=0.050):
    measurement_groups = []
    current_group = []
    base_time = measurements[0][3]

    for measurement in measurements:
        if measurement[3] - base_time <= max_time_diff:
            current_group.append(measurement)
        else:
            measurement_groups.append(current_group)
            current_group = [measurement]
            base_time = measurement[3]

    if current_group:
        measurement_groups.append(current_group)

    return measurement_groups

def form_clusters_via_association(tracks, reports, kalman_filter):
    association_list = []
    cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])  # 3x3 covariance matrix for position only
    chi2_threshold = kalman_filter.gate_threshold

    for i, track in enumerate(tracks):
        for j, report in enumerate(reports):
            distance = mahalanobis_distance(track, report, cov_inv)
            if distance < chi2_threshold:
                association_list.append((i, j))

    clusters = []
    while association_list:
        cluster_tracks = set()
        cluster_reports = set()
        stack = [association_list.pop(0)]

        while stack:
            track_idx, report_idx = stack.pop()
            cluster_tracks.add(track_idx)
            cluster_reports.add(report_idx)
            new_assoc = [(t, r) for t, r in association_list if t == track_idx or r == report_idx]
            for assoc in new_assoc:
                if assoc not in stack:
                    stack.append(assoc)
            association_list = [assoc for assoc in association_list if assoc not in new_assoc]

        clusters.append((list(cluster_tracks), [reports[r] for r in cluster_reports]))

    return clusters

def mahalanobis_distance(track, report, cov_inv):
    residual = np.array(report) - np.array(track)
    distance = np.dot(np.dot(residual.T, cov_inv), residual)
    return distance

def select_best_report(cluster_tracks, cluster_reports, kalman_filter):
    cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])

    best_report = None
    best_track_idx = None
    max_weight = -np.inf

    for i, track in enumerate(cluster_tracks):
        for j, report in enumerate(cluster_reports):
            residual = np.array(report) - np.array(track)
            weight = np.exp(-0.5 * np.dot(np.dot(residual.T, cov_inv), residual))
            if weight > max_weight:
                max_weight = weight
                best_report = report
                best_track_idx = i

    return best_track_idx, best_report

def select_initiation_mode(mode):
    if mode == '3-state':
        return 3
    elif mode == '5-state':
        return 5
    elif mode == '7-state':
        return 7
    else:
        raise ValueError("Invalid mode selected.")

def doppler_correlation(doppler_1, doppler_2, doppler_threshold):
    return abs(doppler_1 - doppler_2) < doppler_threshold

def correlation_check(track, measurement, doppler_threshold, range_threshold):
    last_measurement = track['measurements'][-1][0]
    last_cartesian = sph2cart(last_measurement[0], last_measurement[1], last_measurement[2])
    measurement_cartesian = sph2cart(measurement[0], measurement[1], measurement[2])
    distance = np.linalg.norm(np.array(measurement_cartesian) - np.array(last_cartesian))

    doppler_correlated = doppler_correlation(measurement[4], last_measurement[4], doppler_threshold)
    range_satisfied = distance < range_threshold

    return doppler_correlated and range_satisfied

def initialize_filter_state(kalman_filter, x, y, z, vx, vy, vz, time):
    kalman_filter.initialize_filter_state(x, y, z, vx, vy, vz, time)

def perform_jpda(tracks, reports, kalman_filter):
    clusters = form_clusters_via_association(tracks, reports, kalman_filter)
    best_reports = []
    hypotheses = []
    probabilities = []

    for cluster_tracks, cluster_reports in clusters:
        # Generate hypotheses for each cluster
        cluster_hypotheses = []
        cluster_probabilities = []
        for track in cluster_tracks:
            for report in cluster_reports:
                # Calculate the probability of the hypothesis
                cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])
                residual = np.array(report) - np.array(track)
                probability = np.exp(-0.5 * np.dot(np.dot(residual.T, cov_inv), residual))
                cluster_hypotheses.append((track, report))
                cluster_probabilities.append(probability)

        # Normalize probabilities
        total_probability = sum(cluster_probabilities)
        cluster_probabilities = [p / total_probability for p in cluster_probabilities]

        # Select the best hypothesis based on the highest probability
        best_hypothesis_index = np.argmax(cluster_probabilities)
        best_track, best_report = cluster_hypotheses[best_hypothesis_index]

        # Bias Removal
        bias = np.mean([np.array(report) - np.array(track) for track, report in cluster_hypotheses], axis=0)
        best_report = best_report - bias

        best_reports.append((best_track, best_report))
        hypotheses.append(cluster_hypotheses)
        probabilities.append(cluster_probabilities)

    # Track Coalescence
    coalesced_tracks = []
    for i, (track, report) in enumerate(best_reports):
        merged = False
        for j, (other_track, other_report) in enumerate(coalesced_tracks):
            distance = np.linalg.norm(np.array(track) - np.array(other_track))
            if distance < kalman_filter.gate_threshold:  # Use a suitable threshold
                # Merge tracks
                coalesced_tracks[j] = ((np.array(track) + np.array(other_track)) / 2, (np.array(report) + np.array(other_report)) / 2)
                merged = True
                break
        if not merged:
            coalesced_tracks.append((track, report))

    # Log clusters, hypotheses, and probabilities
    print("JPDA Clusters:", clusters)
    print("JPDA Hypotheses:", hypotheses)
    print("JPDA Probabilities:", probabilities)
    print("JPDA Best Reports:", best_reports)
    print("Coalesced Tracks:", coalesced_tracks)

    return clusters, coalesced_tracks, hypotheses, probabilities

def perform_munkres(tracks, reports, kalman_filter):
    cost_matrix = []
    cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])

    for track in tracks:
        track_costs = []
        for report in reports:
            distance = mahalanobis_distance(track, report, cov_inv)
            track_costs.append(distance)
        cost_matrix.append(track_costs)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    best_reports = [(row, reports[col]) for row, col in zip(row_ind, col_ind)]

    # Log cost matrix and assignments
    print("Munkres Cost Matrix:", cost_matrix)
    print("Munkres Assignments:", list(zip(row_ind, col_ind)))
    print("Munkres Best Reports:", best_reports)

    return best_reports

def check_track_timeout(tracks, current_time, poss_timeout=20.0, firm_tent_timeout=50.0):
    tracks_to_remove = []
    for track_id, track in enumerate(tracks):
        last_measurement_time = track['measurements'][-1][0][3]  # Assuming the time is at index 3
        time_since_last_measurement = current_time - last_measurement_time

        if track['current_state'] == 'Poss1' and time_since_last_measurement > poss_timeout:
            tracks_to_remove.append(track_id)
        elif track['current_state'] in ['Tentative1', 'Firm'] and time_since_last_measurement > firm_tent_timeout:
            tracks_to_remove.append(track_id)

    return tracks_to_remove

def plot_measurements(tracks, ax, plot_type, selected_track_ids=None):
    ax.clear()
    for track in tracks:
        if selected_track_ids is not None and track['track_id'] not in selected_track_ids:
            continue

        times = [m[0][3] for m in track['measurements']]
        measurements_x = [(m[0][:3])[0] for m in track['measurements']]
        measurements_y = [(m[0][:3])[1] for m in track['measurements']]
        measurements_z = [(m[0][:3])[2] for m in track['measurements']]

        # Plot Sf values starting from the third measurement
        if len(track['Sf']) > 2:
            Sf_x = [state[0] for state in track['Sf'][2:]]
            Sf_y = [state[1] for state in track['Sf'][2:]]
            Sf_z = [state[2] for state in track['Sf'][2:]]
            Sf_times = times[2:]
        else:
            Sf_x, Sf_y, Sf_z, Sf_times = [], [], [], []

        if plot_type == "Range vs Time":
            ax.scatter(times, measurements_x, label=f'Track {track["track_id"]} Measurement X', marker='o')
            ax.scatter(Sf_times, Sf_x, label=f'Track {track["track_id"]} Sf X', linestyle='--')
            ax.set_ylabel('X Coordinate')
        elif plot_type == "Azimuth vs Time":
            ax.scatter(times, measurements_y, label=f'Track {track["track_id"]} Measurement Y', marker='o')
            ax.scatter(Sf_times, Sf_y, label=f'Track {track["track_id"]} Sf Y', linestyle='--')
            ax.set_ylabel('Y Coordinate')
        elif plot_type == "Elevation vs Time":
            ax.scatter(times, measurements_z, label=f'Track {track["track_id"]} Measurement Z', marker='o')
            ax.scatter(Sf_times, Sf_z, label=f'Track {track["track_id"]} Sf Z', linestyle='--')
            ax.set_ylabel('Z Coordinate')

    ax.set_xlabel('Time')
    ax.set_title(f'Tracks {plot_type}')
    ax.legend()

    # Add interactive data tips
    cursor = mplcursors.cursor(hover=True)

    @cursor.connect("add")
    def on_add(sel):
        index = sel.target.index
        track_id = tracks[index // len(tracks[0]['measurements'])]['track_id']
        measurement = tracks[index // len(tracks[0]['measurements'])]['measurements'][index % len(tracks[0]['measurements'])]
        time = measurement[0][3]
        sp = tracks[index // len(tracks[0]['measurements'])]['Sp']
        sf = tracks[index // len(tracks[0]['measurements'])]['Sf']
        plant_noise = tracks[index // len(tracks[0]['measurements'])]['Pf'][0, 0]  # Example of accessing plant noise

        sel.annotation.set(text=f"Track ID: {track_id}\nMeasurement: {measurement}\nTime: {time}\nSp: {sp}\nSf: {sf}\nPlant Noise: {plant_noise}")

def log_to_csv(log_file_path, data):
    with open(log_file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writerow(data)

def main(input_file, track_mode, filter_option, association_type):
    log_file_path = 'detailed_log.csv'

    # Initialize CSV log file
    with open(log_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Time', 'Measurement X', 'Measurement Y', 'Measurement Z', 'Current State',
                      'Correlation Output', 'Associated Track ID', 'Associated Position X',
                      'Associated Position Y', 'Associated Position Z', 'Association Type',
                      'Clusters Formed', 'Hypotheses Generated', 'Probability of Hypothesis',
                      'Best Report Selected']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    measurements = read_measurements_from_csv(input_file)

    if filter_option == "CV":
        kalman_filter = CVFilter()
    elif filter_option == "CA":
        kalman_filter = CAFilter()
    else:
        raise ValueError("Invalid filter option selected.")

    measurement_groups = form_measurement_groups(measurements, max_time_diff=0.050)

    tracks = []
    track_id_list = []
    filter_states = []

    doppler_threshold = 100
    range_threshold = 100
    firm_threshold = select_initiation_mode(track_mode)
    association_method = association_type  # 'JPDA' or 'Munkres'

    # Initialize variables outside the loop
    miss_counts = {}
    hit_counts = {}
    firm_ids = set()
    state_map = {}
    state_transition_times = {}
    progression_states = {
        3: ['Poss1', 'Tentative1', 'Firm'],
        5: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Firm'],
        7: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Tentative3', 'Firm']
    }[firm_threshold]

    last_check_time = 0
    check_interval = 0.0005  # 0.5 ms

    for group_idx, group in enumerate(measurement_groups):
        print(f"Processing measurement group {group_idx + 1}...")

        current_time = group[0][3]  # Assuming the time is at index 3 of each measurement

        # Periodic checking
        if current_time - last_check_time >= check_interval:
            tracks_to_remove = check_track_timeout(tracks, current_time)
            for track_id in reversed(tracks_to_remove):
                print(f"Removing track {track_id} due to timeout")
                del tracks[track_id]
                track_id_list[track_id]['state'] = 'free'
                if track_id in firm_ids:
                    firm_ids.remove(track_id)
                if track_id in state_map:
                    del state_map[track_id]
                if track_id in hit_counts:
                    del hit_counts[track_id]
                if track_id in miss_counts:
                    del miss_counts[track_id]
            last_check_time = current_time

        if len(group) == 1:  # Single measurement
            measurement = group[0]
            assigned = False
            for track_id, track in enumerate(tracks):                
                if correlation_check(track, measurement, doppler_threshold, range_threshold):
                    current_state = state_map.get(track_id, None)
                    if current_state == 'Poss1':
                        initialize_filter_state(kalman_filter, *sph2cart(*measurement[:3]), 0, 0, 0, measurement[3])
                    elif current_state == 'Tentative1':
                        last_measurement = track['measurements'][-1][0]
                        dt = measurement[3] - last_measurement[3]
                        vx = (sph2cart(*measurement[:3])[0] - sph2cart(*last_measurement[:3])[0]) / dt
                        vy = (sph2cart(*measurement[:3])[1] - sph2cart(*last_measurement[:3])[1]) / dt
                        vz = (sph2cart(*measurement[:3])[2] - sph2cart(*last_measurement[:3])[2]) / dt
                        initialize_filter_state(kalman_filter, *sph2cart(*measurement[:3]), vx, vy, vz, measurement[3])
                    elif current_state == 'Firm':
                        kalman_filter.predict_step(measurement[3])
                        kalman_filter.update_step(np.array((measurement[:3])).reshape(3, 1))

                    track['measurements'].append((measurement, current_state))
                    track['Sf'].append(kalman_filter.Sf.copy())
                    track['Sp'].append(kalman_filter.Sp.copy())
                    track['Pp'].append(kalman_filter.Pp.copy())
                    track['Pf'].append(kalman_filter.Pf.copy())
                    hit_counts[track_id] = hit_counts.get(track_id, 0) + 1
                    assigned = True

                    # Log data to CSV
                    log_data = {
                        'Time': measurement[3],
                        'Measurement X': measurement[5],
                        'Measurement Y': measurement[6],
                        'Measurement Z': measurement[7],
                        'Current State': current_state,
                        'Correlation Output': 'Yes',
                        'Associated Track ID': track_id,
                        'Associated Position X': track['Sf'][-1][0, 0],
                        'Associated Position Y': track['Sf'][-1][1, 0],
                        'Associated Position Z': track['Sf'][-1][2, 0],
                        'Association Type': 'Single',
                        'Clusters Formed': '',
                        'Hypotheses Generated': '',
                        'Probability of Hypothesis': '',
                        'Best Report Selected': ''
                    }
                    log_to_csv(log_file_path, log_data)
                    break

            if not assigned:
                new_track_id = next((i for i, t in enumerate(track_id_list) if t['state'] == 'free'), None)
                if new_track_id is None:
                    new_track_id = len(track_id_list)
                    track_id_list.append({'id': new_track_id, 'state': 'occupied'})
                else:
                    track_id_list[new_track_id]['state'] = 'occupied'

                tracks.append({
                    'track_id': new_track_id,
                    'measurements': [(measurement, 'Poss1')],
                    'current_state': 'Poss1',
                    'Sf': [kalman_filter.Sf.copy()],
                    'Sp': [kalman_filter.Sp.copy()],
                    'Pp': [kalman_filter.Pp.copy()],
                    'Pf': [kalman_filter.Pf.copy()]
                })
                state_map[new_track_id] = 'Poss1'
                state_transition_times[new_track_id] = {'Poss1': current_time}
                hit_counts[new_track_id] = 1
                initialize_filter_state(kalman_filter, *sph2cart(*measurement[:3]), 0, 0, 0, measurement[3])

                # Log data to CSV
                log_data = {
                    'Time': measurement[3],
                    'Measurement X': measurement[5],
                    'Measurement Y': measurement[6],
                    'Measurement Z': measurement[7],
                    'Current State': 'Poss1',
                    'Correlation Output': 'No',
                    'Associated Track ID': new_track_id,
                    'Associated Position X': '',
                    'Associated Position Y': '',
                    'Associated Position Z': '',
                    'Association Type': 'New',
                    'Clusters Formed': '',
                    'Hypotheses Generated': '',
                    'Probability of Hypothesis': '',
                    'Best Report Selected': ''
                }
                log_to_csv(log_file_path, log_data)

        else:  # Multiple measurements
            reports = [sph2cart(*m[:3]) for m in group]
            if association_method == 'JPDA':
                clusters, best_reports, hypotheses, probabilities = perform_jpda(
                    [track['measurements'][-1][0][:3] for track in tracks], reports, kalman_filter
                )
            elif association_method == 'Munkres':
                best_reports = perform_munkres([track['measurements'][-1][0][:3] for track in tracks], reports, kalman_filter)

            for track_id, best_report in best_reports:
                print("check the best reports",)
                current_state = state_map.get(track_id, None)
                if current_state == 'Poss1':
                    initialize_filter_state(kalman_filter, *best_report, vx, vy, vz, group[0][3])
                elif current_state == 'Tentative1':
                    last_measurement = tracks[track_id]['measurements'][-1][0]
                    dt = group[0][3] - last_measurement[3]
                    vx = (best_report[0] - sph2cart(*last_measurement[:3])[0]) / dt
                    vy = (best_report[1] - sph2cart(*last_measurement[:3])[1]) / dt
                    vz = (best_report[2] - sph2cart(*last_measurement[:3])[2]) / dt
                    initialize_filter_state(kalman_filter, *best_report, vx, vy, vz, group[0][3])
                elif current_state == 'Firm':
                    kalman_filter.predict_step(group[0][3])
                    kalman_filter.update_step(np.array(best_report).reshape(3, 1))

                tracks[track_id]['measurements'].append((cart2sph(*best_report) + (group[0][3], group[0][4]), current_state))
                tracks[track_id]['Sf'].append(kalman_filter.Sf.copy())
                tracks[track_id]['Sp'].append(kalman_filter.Sp.copy())
                tracks[track_id]['Pp'].append(kalman_filter.Pp.copy())
                tracks[track_id]['Pf'].append(kalman_filter.Pf.copy())
                hit_counts[track_id] = hit_counts.get(track_id, 0) + 1

                # Log data to CSV
                log_data = {
                    'Time': group[0][3],
                    'Measurement X': best_report[0],
                    'Measurement Y': best_report[1],
                    'Measurement Z': best_report[2],
                    'Current State': current_state,
                    'Correlation Output': 'Yes',
                    'Associated Track ID': track_id,
                    'Associated Position X': tracks[track_id]['Sf'][-1][0, 0],
                    'Associated Position Y': tracks[track_id]['Sf'][-1][1, 0],
                    'Associated Position Z': tracks[track_id]['Sf'][-1][2, 0],
                    'Association Type': association_method,
                    'Hypotheses Generated': '',
                    'Probability of Hypothesis': '',
                    'Best Report Selected': best_report
                }
                log_to_csv(log_file_path, log_data)

            # Handle unassigned measurements
            assigned_reports = set(best_report for _, best_report in best_reports)
            for report in reports:
                if tuple(report) not in assigned_reports:
                    new_track_id = next((i for i, t in enumerate(track_id_list) if t['state'] == 'free'), None)
                    if new_track_id is None:
                        new_track_id = len(track_id_list)
                        track_id_list.append({'id': new_track_id, 'state': 'occupied'})
                    else:
                        track_id_list[new_track_id]['state'] = 'occupied'

                    tracks.append({
                        'track_id': new_track_id,
                        'measurements': [(cart2sph(*report) + (group[0][3], group[0][4]), 'Poss1')],
                        'current_state': 'Poss1',
                        'Sf': [kalman_filter.Sf.copy()],
                        'Sp': [kalman_filter.Sp.copy()],
                        'Pp': [kalman_filter.Pp.copy()],
                        'Pf': [kalman_filter.Pf.copy()]
                    })
                    state_map[new_track_id] = 'Poss1'
                    state_transition_times[new_track_id] = {'Poss1': current_time}
                    hit_counts[new_track_id] = 1
                    initialize_filter_state(kalman_filter, *report, 0, 0, 0, group[0][3])

                    # Log data to CSV
                    log_data = {
                        'Time': group[0][3],
                        'Measurement X': report[0],
                        'Measurement Y': report[1],
                        'Measurement Z': report[2],
                        'Current State': 'Poss1',
                        'Correlation Output': 'No',
                        'Associated Track ID': new_track_id,
                        'Associated Position X': '',
                        'Associated Position Y': '',
                        'Associated Position Z': '',
                        'Association Type': 'New',
                        'Hypotheses Generated': '',
                        'Probability of Hypothesis': '',
                        'Best Report Selected': ''
                    }
                    log_to_csv(log_file_path, log_data)

        # Update states based on hit counts
        for track_id, track in enumerate(tracks):
            current_state = state_map.get(track_id,None)
            if current_state is not None:
                current_state_index = progression_states.index(current_state)
                if hit_counts[track_id] >= firm_threshold and current_state != 'Firm':
                    state_map[track_id] = 'Firm'
                    firm_ids.add(track_id)
                    state_transition_times.setdefault(track_id, {})['Firm'] = current_time
                elif current_state_index < len(progression_states) - 1:
                    next_state = progression_states[current_state_index + 1]
                    if hit_counts[track_id] >= current_state_index + 1 and state_map[track_id] != next_state:
                        state_map[track_id] = next_state
                        state_transition_times.setdefault(track_id, {})[next_state] = current_time
                track['current_state'] = state_map[track_id]

    # Prepare data for CSV
    csv_data = []
    for track_id, track in enumerate(tracks):
        print(f"Track {track_id}:")
        print(f"  Current State: {track['current_state']}")
        print(f"  State Transition Times:")
        for state, time in state_transition_times.get(track_id, {}).items():
            print(f"    {state}: {time}")
        print("  Measurement History:")
        for state in progression_states:
            measurements = [m for m, s in track['measurements'] if s == state][:3]
            print(f"    {state}: {measurements}")
        print(f"  Track Status: {track_id_list[track_id]['state']}")
        print(f"  SF: {track['Sf']}")
        print(f"  SP: {track['Sp']}")
        print(f"  PF: {track['Pf']}")
        print(f"  PP: {track['Pp']}")
        print()

        # Prepare data for CSV
        csv_data.append({
            'Track ID': track_id,
            'Current State': track['current_state'],
            'Poss1 Time': state_transition_times.get(track_id, {}).get('Poss1', ''),
            'Tentative1 Time': state_transition_times.get(track_id, {}).get('Tentative1', ''),
            'Firm Time': state_transition_times.get(track_id, {}).get('Firm', ''),
            'Poss1 Measurements': str([m for m, s in track['measurements'] if s == 'Poss1'][:3]),
            'Tentative1 Measurements': str([m for m, s in track['measurements'] if s == 'Tentative1'][:3]),
            'Firm Measurements': str([m for m, s in track['measurements'] if s == 'Firm'][:3]),
            'Track Status': track_id_list[track_id]['state'],
            'SF': [sf.tolist() for sf in track['Sf']],
            'SP': [sp.tolist() for sp in track['Sp']],
            'PF': [pf.tolist() for pf in track['Pf']],
            'PP': [pp.tolist() for pp in track['Pp']]
        })

    # Write to CSV
    csv_file_path = 'track_summary.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Track ID', 'Current State', 'Poss1 Time', 'Tentative1 Time', 'Firm Time',
                      'Poss1 Measurements', 'Tentative1 Measurements', 'Firm Measurements',
                      'Track Status', 'SF', 'SP', 'PF', 'PP']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)

    print(f"Track summary has been written to {csv_file_path}")

    # Add this line at the end of the function
    return tracks

class SystemConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Configuration")
        self.setGeometry(300, 300, 350, 350)

        grid = QGridLayout()

        # Target Speed
        self.target_speed_group = QGroupBox("Target Speed (m/s)")
        speed_layout = QHBoxLayout()
        self.min_speed_edit = QLineEdit()
        self.min_speed_edit.setPlaceholderText("Min")
        speed_layout.addWidget(self.min_speed_edit)
        self.max_speed_edit = QLineEdit()
        self.max_speed_edit.setPlaceholderText("Max")
        speed_layout.addWidget(self.max_speed_edit)
        self.target_speed_group.setLayout(speed_layout)
        grid.addWidget(self.target_speed_group, 0, 0, 1, 2)

        # Target Altitude
        self.target_altitude_group = QGroupBox("Target Altitude (m)")
        altitude_layout = QHBoxLayout()
        self.min_altitude_edit = QLineEdit()
        self.min_altitude_edit.setPlaceholderText("Min")
        altitude_layout.addWidget(self.min_altitude_edit)
        self.max_altitude_edit = QLineEdit()
        self.max_altitude_edit.setPlaceholderText("Max")
        altitude_layout.addWidget(self.max_altitude_edit)
        self.target_altitude_group.setLayout(altitude_layout)
        grid.addWidget(self.target_altitude_group, 1, 0, 1, 2)

        # Correlation Gates
        self.correlation_gates_group = QGroupBox("Correlation Gates")
        gates_layout = QGridLayout()
        self.range_gate_group = QGroupBox("Range Gate (m)")
        range_layout = QHBoxLayout()
        self.min_range_edit = QLineEdit()
        self.min_range_edit.setPlaceholderText("Min")
        range_layout.addWidget(self.min_range_edit)
        self.max_range_edit = QLineEdit()
        self.max_range_edit.setPlaceholderText("Max")
        range_layout.addWidget(self.max_range_edit)
        self.range_gate_group.setLayout(range_layout)
        gates_layout.addWidget(self.range_gate_group, 0, 0)

        self.azimuth_gate_group = QGroupBox("Azimuth Gate (°)")
        azimuth_layout = QHBoxLayout()
        self.min_azimuth_edit = QLineEdit()
        self.min_azimuth_edit.setPlaceholderText("Min")
        azimuth_layout.addWidget(self.min_azimuth_edit)
        self.max_azimuth_edit = QLineEdit()
        self.max_azimuth_edit.setPlaceholderText("Max")
        azimuth_layout.addWidget(self.max_azimuth_edit)
        self.azimuth_gate_group.setLayout(azimuth_layout)
        gates_layout.addWidget(self.azimuth_gate_group, 1, 0)

        self.elevation_gate_group = QGroupBox("Elevation Gate (°)")
        elevation_layout = QHBoxLayout()
        self.min_elevation_edit = QLineEdit()
        self.min_elevation_edit.setPlaceholderText("Min")
        elevation_layout.addWidget(self.min_elevation_edit)
        self.max_elevation_edit = QLineEdit()
        self.max_elevation_edit.setPlaceholderText("Max")
        elevation_layout.addWidget(self.max_elevation_edit)
        self.elevation_gate_group.setLayout(elevation_layout)
        gates_layout.addWidget(self.elevation_gate_group, 2, 0)

        self.correlation_gates_group.setLayout(gates_layout)
        grid.addWidget(self.correlation_gates_group, 2, 0, 3, 2)

        # Plant Noise
        self.plant_noise_label = QLabel("Plant Noise Covariance:")
        self.plant_noise_edit = QLineEdit()
        grid.addWidget(self.plant_noise_label, 5, 0)
        grid.addWidget(self.plant_noise_edit, 5, 1)

        # OK and Cancel buttons
        button_box = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_box.addWidget(ok_button)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_box.addWidget(cancel_button)
        grid.addLayout(button_box, 6, 0, 1, 2)

        self.setLayout(grid)

    def get_config_data(self):
        return {
            "target_speed": (float(self.min_speed_edit.text()), float(self.max_speed_edit.text())),
            "target_altitude": (float(self.min_altitude_edit.text()), float(self.max_altitude_edit.text())),
            "range_gate": (float(self.min_range_edit.text()), float(self.max_range_edit.text())),
            "azimuth_gate": (float(self.min_azimuth_edit.text()), float(self.max_azimuth_edit.text())),
            "elevation_gate": (float(self.min_elevation_edit.text()), float(self.max_elevation_edit.text())),
            "plant_noise": float(self.plant_noise_edit.text())
        }

class PlotConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Configuration")
        self.setGeometry(300, 300, 350, 350)

        grid = QGridLayout()

        # Range
        self.range_group = QGroupBox("Range")
        range_layout = QHBoxLayout()
        self.min_range_edit = QLineEdit()
        self.min_range_edit.setPlaceholderText("Min")
        range_layout.addWidget(self.min_range_edit)
        self.max_range_edit = QLineEdit()
        self.max_range_edit.setPlaceholderText("Max")
        range_layout.addWidget(self.max_range_edit)
        self.range_group.setLayout(range_layout)
        grid.addWidget(self.range_group, 0, 0, 1, 2)

        # Azimuth
        self.azimuth_group = QGroupBox("Azimuth")
        azimuth_layout = QHBoxLayout()
        self.min_azimuth_edit = QLineEdit()
        self.min_azimuth_edit.setPlaceholderText("Min")
        azimuth_layout.addWidget(self.min_azimuth_edit)
        self.max_azimuth_edit = QLineEdit()
        self.max_azimuth_edit.setPlaceholderText("Max")
        azimuth_layout.addWidget(self.max_azimuth_edit)
        self.azimuth_group.setLayout(azimuth_layout)
        grid.addWidget(self.azimuth_group, 1, 0, 1, 2)

        # Elevation
        self.elevation_group = QGroupBox("Elevation")
        elevation_layout = QHBoxLayout()
        self.min_elevation_edit = QLineEdit()
        self.min_elevation_edit.setPlaceholderText("Min")
        elevation_layout.addWidget(self.min_elevation_edit)
        self.max_elevation_edit = QLineEdit()
        self.max_elevation_edit.setPlaceholderText("Max")
        elevation_layout.addWidget(self.max_elevation_edit)
        self.elevation_group.setLayout(elevation_layout)
        grid.addWidget(self.elevation_group, 2, 0, 1, 2)

        # Time
        self.time_group = QGroupBox("Time")
        time_layout = QHBoxLayout()
        self.min_time_edit = QLineEdit()
        self.min_time_edit.setPlaceholderText("Min")
        time_layout.addWidget(self.min_time_edit)
        self.max_time_edit = QLineEdit()
        self.max_time_edit.setPlaceholderText("Max")
        time_layout.addWidget(self.max_time_edit)
        self.time_group.setLayout(time_layout)
        grid.addWidget(self.time_group, 3, 0, 1, 2)

        # OK and Cancel buttons
        button_box = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_box.addWidget(ok_button)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_box.addWidget(cancel_button)
        grid.addLayout(button_box, 4, 0, 1, 2)

        self.setLayout(grid)

    def get_config_data(self):
        return {
            "range": (float(self.min_range_edit.text()), float(self.max_range_edit.text())),
            "azimuth": (float(self.min_azimuth_edit.text()), float(self.max_azimuth_edit.text())),
            "elevation": (float(self.min_elevation_edit.text()), float(self.max_elevation_edit.text())),
            "time": (float(self.min_time_edit.text()), float(self.max_time_edit.text()))
        }

class Signal(QObject):
    # Signal for collapsing the control panel
    collapseSignal = pyqtSignal(bool)

class KalmanFilterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.tracks = []
        self.selected_track_ids = set()
        self.marker_size = 10  # Default marker size
        self.plot_color = 'b'  # Default plot color
        self.initUI()
        self.control_panel_collapsed = False  # Start with the panel expanded

    def initUI(self):
        self.setWindowTitle('Kalman Filter GUI')
        self.setGeometry(100, 100, 1200, 600)
        self.setStyleSheet("""
            QWidget {
                background-color: #222222;
                color: #ffffff;
                font-family: "Arial", sans-serif;
            }
            QPushButton {
                background-color: #4CAF50; 
                color: white;
                border: none;
                padding: 8px 16px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3e8e41;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
            }
            QComboBox {
                background-color: #222222;
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
                font-size: 12px;
            }
            QLineEdit {
                background-color: #333333;
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
                font-size: 12px;
            }
            QRadioButton {
                background-color: transparent;
                color: white;
            }
            QTextEdit {
                background-color: #333333;
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
                font-size: 12px;
            }
            QGroupBox {
                background-color: #333333;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
            }
            QTableWidget {
                background-color: #333333;
                color: white;
                border: 1px solid #555555;
                font-size: 12px;
            }
        """)

        # Main layout
        main_layout = QHBoxLayout()

        # Left side: System Configuration and Controls (Collapsible)
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout)

        # Collapse/Expand Button
        self.collapse_button = QToolButton()
        self.collapse_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.collapse_button.setText("=")  # Set the button text to "="
        self.collapse_button.clicked.connect(self.toggle_control_panel)
        left_layout.addWidget(self.collapse_button)
        # Control Panel
        self.control_panel = QWidget()
        self.control_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        control_layout = QVBoxLayout()
        self.control_panel.setLayout(control_layout)
        left_layout.addWidget(self.control_panel)
        # File Upload Button
        self.file_upload_button = QPushButton("Upload File")
        self.file_upload_button.setIcon(QIcon("upload.png"))
        self.file_upload_button.clicked.connect(self.select_file)
        control_layout.addWidget(self.file_upload_button)
        # System Configuration button
        self.config_button = QPushButton("System Configuration")
        self.config_button.setIcon(QIcon("config.png"))
        self.config_button.clicked.connect(self.show_config_dialog)
        control_layout.addWidget(self.config_button)
        # Initiate Track drop down
        self.track_mode_label = QLabel("Initiate Track")
        self.track_mode
