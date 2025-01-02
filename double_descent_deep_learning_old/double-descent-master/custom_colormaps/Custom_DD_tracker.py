import numpy as np

def detect_double_descent_with_local_extrema(ks, mlist):
    """
    Detects a double descent pattern based on local minima and maxima and scores it based on timing of each phase.
    
    Parameters:
    - ks: List of model complexity levels (model sizes).
    - mlist: List of dictionaries containing 'Test Error' for each complexity level.

    Returns:
    - is_double_descent: Boolean indicating if the pattern matches double descent.
    - score: A score indicating how strong the double descent pattern is.
    """
    complexities = np.array(ks)
    test_errors = np.array([np.mean(entry['Test Error']) for entry in mlist])

    # Initialize score and detection variables
    score = 0
    is_double_descent = False

    # Step 1: Identify the first local minimum (initial descent)
    for i in range(1, len(test_errors) - 1):
        if test_errors[i] < test_errors[i - 1] and test_errors[i] < test_errors[i + 1]:
            first_min_index = i
            score += (len(complexities) - i) / len(complexities)  # Early minimum contributes more
            break
    else:
        return is_double_descent, score  # No local minimum found, so no double descent

    # Step 2: Identify the first local maximum after the first minimum
    for j in range(first_min_index + 1, len(test_errors) - 1):
        if test_errors[j] > test_errors[j - 1] and test_errors[j] > test_errors[j + 1]:
            first_max_index = j
            score += (len(complexities) - j) / len(complexities)  # Early maximum contributes more
            break
    else:
        return is_double_descent, score  # No local maximum found after minimum, so no double descent

    # Step 3: Check for monotonic decrease after the local maximum
    if all(test_errors[first_max_index:] <= test_errors[first_max_index]):
        # Check if the final test error is lower than the initial minimum
        final_test_error = test_errors[-1]
        if final_test_error < test_errors[first_min_index]:
            score += 1  # Final minimum being lower than initial minimum increases score
            is_double_descent = True

    return is_double_descent, score

# Run the local extrema-based double descent detection function
is_double_descent, score = detect_double_descent_with_local_extrema(data_ks, data_mlist)

is_double_descent, score
