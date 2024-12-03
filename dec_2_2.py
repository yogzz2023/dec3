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

        # Bias Removal: Adjust the state estimates
        mean_residual = np.mean([np.array(report) - np.array(track) for track, report in cluster_hypotheses], axis=0)
        best_report = best_report - mean_residual

        best_reports.append((best_track, best_report))
        hypotheses.append(cluster_hypotheses)
        probabilities.append(cluster_probabilities)

    # Track Coalescence: Merge tracks that are too close
    coalesced_tracks = []
    for i, (track1, report1) in enumerate(best_reports):
        for j, (track2, report2) in enumerate(best_reports):
            if i != j:
                distance = mahalanobis_distance(track1, track2, np.linalg.inv(kalman_filter.Pp[:3, :3]))
                if distance < kalman_filter.gate_threshold:
                    # Merge tracks
                    merged_track = (np.array(track1) + np.array(track2)) / 2
                    coalesced_tracks.append((merged_track, report1))
                else:
                    coalesced_tracks.append((track1, report1))

    # Log clusters, hypotheses, and probabilities
    print("JPDA Clusters:", clusters)
    print("JPDA Hypotheses:", hypotheses)
    print("JPDA Probabilities:", probabilities)
    print("JPDA Best Reports:", best_reports)
    print("Coalesced Tracks:", coalesced_tracks)

    return clusters, coalesced_tracks, hypotheses, probabilities


    !!!!!!!!!!!!!!
    
    To integrate the updated `perform_jpda` function into your existing code, you need to ensure that it is called in the appropriate places where JPDA (Joint Probabilistic Data Association) is required. Here are the steps and places where you should consider calling this function:

        1. **Replace Existing JPDA Calls**: 
           - Locate where the original `perform_jpda` function is called in your code. This is typically within the main processing loop where measurements are being associated with tracks.
           - Replace these calls with the updated `perform_jpda` function.
        
        2. **Main Processing Loop**:
           - In your `main` function, you have a section where you handle multiple measurements. This is where you decide between using JPDA or Munkres for data association. Ensure that the updated `perform_jpda` function is called here when the JPDA method is selected.
        
        3. **Example Integration**:
           - Here's a snippet showing how you might integrate the updated function within your main loop:
        
        ```python
        def main(input_file, track_mode, filter_option, association_type):
            # ... [existing code] ...
        
            for group_idx, group in enumerate(measurement_groups):
                print(f"Processing measurement group {group_idx + 1}...")
        
                current_time = group[0][3]  # Assuming the time is at index 3 of each measurement
        
                # ... [existing code for single measurement handling] ...
        
                else:  # Multiple measurements
                    reports = [sph2cart(*m[:3]) for m in group]
                    if association_method == 'JPDA':
                        clusters, best_reports, hypotheses, probabilities = perform_jpda(
                            [track['measurements'][-1][0][:3] for track in tracks], reports, kalman_filter
                        )
                    elif association_method == 'Munkres':
                        best_reports = perform_munkres([track['measurements'][-1][0][:3] for track in tracks], reports, kalman_filter)
        
                    # ... [existing code for handling best reports] ...
        ```
        
        4. **Testing**:
           - After integrating the updated function, thoroughly test your application to ensure that the bias removal and track coalescence are functioning as expected. Check the logs and outputs to verify that the tracks are being processed correctly.
        
        5. **Adjustments**:
           - Depending on the specific requirements of your application, you might need to adjust the parameters or logic within the `perform_jpda` function, especially the thresholds used for bias removal and track coalescence.
        
        By following these steps, you should be able to integrate the updated JPDA function into your existing codebase effectively.