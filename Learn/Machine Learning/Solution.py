def DecisionTree():
    #      This function should return a list of labels.
    #      e.g.:
    #	labels = [['+','-','+'],['+','+','-'],['-','+'],['+','+']]
    #	return labels
    #	where:
    #		labels[0] = original_training_labels
    #		labels[1] = prediected_training_labels
    #		labels[2] = original_testing_labels
    #		labels[3] = predicted_testing_labels
    attribute_list = list(range(0, 15))
    max_depth = len(attribute_list)
    labels = DecisionTreeBounded(max_depth)
    return labels


def predict(row, root):
    class_label = None
    if (root is None or row is None):
        return None

    if (isinstance(root, dict)):
        if (root['type'] == 'C'):
            if (float(row[root['attribute']]) <= root['threshold']):
                class_label = predict(row, root['left'])
            else:
                class_label = predict(row, root['right'])
        else:
            if (row[root['attribute']] in root['groups']):
                class_label = predict(row, root[row[root['attribute']]])
    else:
        return root
    return class_label


def predict_dataset(data, root):
    output = list()
    for row in data:
        output.append(predict(row, root))
    return output


def get_dataset_class_labels(data):
    return list(row[-1] for row in data)


def impute_missing_data(data, attribute_info):
    configure_imputed_value(data, attribute_info)
    for row in data:
        for attribute in attribute_info['column_number']:
            if (row[attribute] == '?'):
                row[attribute] = attribute_info['imputed_value'][attribute]
    return


def configure_imputed_value(data, attribute_info):
    # Using median for continuous and mode for categorical variables
    attribute_info['imputed_value'] = [[] for i in range(len(attribute_info['column_number']))]
    for row in data:
        for attribute in attribute_info['column_number']:
            if (row[attribute] != '?'):
                if (attribute_info['type'][attribute] == 'C'):
                    attribute_info['imputed_value'][attribute].append(float(row[attribute].strip()))
                else:
                    attribute_info['imputed_value'][attribute].append(row[attribute])

    import statistics
    # find median and mode for each list
    for attribute in attribute_info['column_number']:
        if (attribute_info['type'][attribute] == 'C'):
            attribute_info['imputed_value'][attribute] = statistics.median(attribute_info['imputed_value'][attribute])
        else:
            attribute_info['imputed_value'][attribute] = statistics.mode(attribute_info['imputed_value'][attribute])
    return


def configure_attribute_info(attribute_list):
    attribute_info = {}
    attribute_info['column_number'] = attribute_list.copy()
    # Type contains C-Continuous, D-Discreet
    attribute_info['type'] = list(['D', 'C', 'C', 'D', 'D', 'D', 'D', 'C', 'D', 'D', 'C', 'D', 'D', 'C', 'C'])
    # Available Discreet value list. Configuring 0 for continuous values
    attribute_info['values'] = [['b', 'a'], 0, 0, ['u', 'y', 'l', 't'], ['g', 'p', 'gg'],
                                ['c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff'],
                                ['v', 'h', 'bb', 'j', 'n', 'z', 'dd', 'ff', 'o'], 0, ['t', 'f'], ['t', 'f'],
                                0, ['t', 'f'], ['g', 'p', 's'], 0, 0]

    return attribute_info


def calculate_entropy(data):
    """
    Calculates entropy of the given data
    :param data: type - list
    :return: entropy
    """
    import math
    class_labels = get_class_labels_list(data)
    entropy = 0
    for class_label in class_labels:
        class_label_probability = count_class_label_occurence(data, class_label) / len(data)
        entropy = entropy - class_label_probability * math.log2(class_label_probability)
    return entropy


def get_class_labels_list(data):
    return list(set(row[-1] for row in data))


def count_class_label_occurence(data, class_label):
    return list(row[-1] for row in data).count(class_label)


def calculate_information_gain(data, attribute, attribute_info):
    """
    Calculates information gain for given attribute (column number)
    :param attribute_info: Dict
    :param data: dataset consisting of all attributes
    :param attribute: column number
    :return: information gain value, constructed node
    """
    if (len(data) == 0):
        return 0, None
    information_gain = 0
    node = {}
    node['attribute'] = attribute_info['column_number'][attribute]
    node['type'] = attribute_info['type'][attribute]

    entropy = calculate_entropy(data)
    # For discreet - Need to partition data into two subgroups and then find information_gain
    if attribute_info['type'][attribute] == 'D':

        attribute_label_list, probability_list, expected_entropy_list = calculate_expected_entropy_discreet(
            data, attribute, attribute_info['values'][attribute])
        if (sum(expected_entropy_list) == 0):
            return 0, None
        information_gain = entropy - sum([a * b for a, b in zip(expected_entropy_list, probability_list)])

        left_group, right_group = group_discreet_attribute_values(attribute_label_list, probability_list,
                                                                  expected_entropy_list)
        node['left_group'] = left_group
        node['right_group'] = right_group

    # For continuous - Need to find threshold and then information_gain
    else:
        threshold = find_continuous_attribute_threshold(data, attribute)
        information_gain = entropy - calculate_expected_entropy_continuous(data, attribute, threshold)
        node['threshold'] = threshold

    return information_gain, node


def find_continuous_attribute_threshold(data, attribute):
    threshold = 0
    list_attributevalue_class_label = list([float(row[attribute]), row[-1]] for row in data)
    list_attributevalue_class_label = sorted(list_attributevalue_class_label, key=lambda x: x[0])
    class_labels = get_class_labels_list(data)
    bins = [0] * len(class_labels)
    max_proportion = 0
    index = 0
    for i in range(len(list_attributevalue_class_label)):
        bins[class_labels.index(list_attributevalue_class_label[i][1])] += 1
        if (sum(bins) - bins[0] != 0):
            if (max_proportion < bins[0] / (sum(bins) - bins[0])):
                max_proportion = bins[0] / (sum(bins) - bins[0])
                index = i
    threshold = list_attributevalue_class_label[index][0]
    return threshold


def calculate_expected_entropy_continuous(data, attribute, threshold):
    expected_entropy = 0
    # find count of dataset
    data_count = len(data)
    # Count of items with values less than threshold
    count_lt_threshold = 0
    class_labels = get_class_labels_list(data)
    lt_threshold_count_per_class_label_list = [0] * len(class_labels)
    gt_threshold_count_per_class_label_list = [0] * len(class_labels)

    for row in data:
        if (float(row[attribute]) < float(threshold)):
            count_lt_threshold += 1
            lt_threshold_count_per_class_label_list[class_labels.index(row[-1])] += 1
        else:
            gt_threshold_count_per_class_label_list[class_labels.index(row[-1])] += 1

    # Count of items with values greater than threshold
    count_gt_threshold = data_count - count_lt_threshold
    import math
    expected_entropy_lt = 0
    expected_entropy_gt = 0
    for i in range(len(class_labels)):
        if (lt_threshold_count_per_class_label_list[i] != 0):
            expected_entropy_lt = expected_entropy_lt - (
                        lt_threshold_count_per_class_label_list[i] / count_lt_threshold) * (
                                      math.log2(lt_threshold_count_per_class_label_list[i] / count_lt_threshold))
        if (gt_threshold_count_per_class_label_list[i] != 0):
            expected_entropy_gt = expected_entropy_gt - (
                        gt_threshold_count_per_class_label_list[i] / count_gt_threshold) * (
                                      math.log2(gt_threshold_count_per_class_label_list[i] / count_gt_threshold))
    expected_entropy = expected_entropy_lt * (count_lt_threshold / data_count) + expected_entropy_gt * (
                count_gt_threshold / data_count)
    return expected_entropy


def calculate_expected_entropy_discreet(data, attribute, values):
    expected_entropy_list = list()
    attribute_label_list = list()
    probability_list = list()
    for attribute_label in values:

        # find count of dataset
        data_count = len(data)
        if (data_count == 0):  # No data present
            continue
        # find count of attribute_label in dataset
        attribute_label_count = list(row[attribute] for row in data).count(attribute_label)
        # above two used in probability
        probability_attribute_label_in_data = attribute_label_count / data_count
        # append to probability_list
        probability_list.append(probability_attribute_label_in_data)

        # find count of attribute_label associated to each class_label
        class_labels = get_class_labels_list(data)
        attribute_label_count_per_class_label_list = [0] * len(class_labels)
        for row in data:
            if (row[attribute] == attribute_label and row[-1] in class_labels):
                attribute_label_count_per_class_label_list[class_labels.index(row[-1])] += 1

        # find prob using above and count of attribute_label in dataset
        # Calculate entropy of this list
        expected_entropy = 0
        import math
        for i in range(len(class_labels)):
            if (attribute_label_count_per_class_label_list[i] != 0):
                expected_entropy = expected_entropy - (
                            attribute_label_count_per_class_label_list[i] / attribute_label_count) * math.log2(
                    attribute_label_count_per_class_label_list[i] / attribute_label_count)
        # append to expected_entropy_list
        expected_entropy_list.append(expected_entropy)
        attribute_label_list.append(attribute_label)

    return attribute_label_list, probability_list, expected_entropy_list


def group_discreet_attribute_values(attribute_label_list, probability_list, expected_entropy_list):
    """
    Divide attribute_label_list into two groups with reasonable similar expected entropy
    :param attribute_label_list:
    :param probability_list:
    :param expected_entropy_list:
    :return:
    """
    left_group = list()
    right_group = list()
    left_sum = 0
    right_sum = 0
    for i in range(len(attribute_label_list)):
        if (left_sum <= right_sum):
            left_group.append(attribute_label_list[i])
            left_sum += probability_list[i] * expected_entropy_list[i]
        else:
            right_group.append(attribute_label_list[i])
            right_sum += probability_list[i] * expected_entropy_list[i]

    return left_group, right_group


def get_attribute_for_split(data, attribute_list, root, attribute_info):
    column = None
    information_gain = 0
    root_copy = None
    if (len(attribute_list) == 0):
        return None
    else:
        # Calculate Information Gain for each attribute
        for attribute in attribute_list:
            attribute_information_gain, root_copy = calculate_information_gain(data, attribute, attribute_info)
            if (information_gain <= attribute_information_gain):
                information_gain = attribute_information_gain
                root = root_copy
    return root


def build_tree(data, root, attribute_list, current_depth, max_depth, attribute_info):
    if (not isinstance(root, dict)):
        return root

    node = get_attribute_for_split(data, attribute_list, root, attribute_info)
    if (node is None):
        return to_terminal(data)

    #Create children
    if(node['type']=='D'):
        node['groups'] = attribute_info['values'][node['attribute']]
    else:
        node['groups'] = ['left','right']

    attribute_list_copy = attribute_list.copy()
    attribute_list_copy.pop(attribute_list.index(node['attribute']))

    # Splitting dataset to left,right subset
    groups_subtree = split_dataset(node['attribute'], node, data)

    if node['type']=='C':
        if (len(groups_subtree[0]) == 0):
            return to_terminal(groups_subtree[1])
        elif (len(groups_subtree[1]) == 0):
            return to_terminal(groups_subtree[0])

    if (current_depth >= max_depth):
        for i in range(len(node['groups'])):
            if len(groups_subtree[i]) != 0:
                node[node['groups'][i]] = to_terminal(groups_subtree[i])
            else:
                node[node['groups'][i]] = get_class_labels_list(data)[0]#Guess the value if not data available during training

    else:
        for i in range(len(node['groups'])):
            if len(groups_subtree[i]) != 0:
                node[node['groups'][i]] = build_tree(groups_subtree[i],
                                                node, attribute_list_copy, current_depth + 1, max_depth, attribute_info)
            else:
                node[node['groups'][i]] = get_class_labels_list(data)[0]#Guess the value if not data available during training

    return node


def split_dataset(index, root, data):
    # tree
    groups_subtree = [[] for i in range(len(root['groups']))]
    left, right = list(), list()

    if (root['type'] == 'C'):
        for row in data:
            if float(row[index]) < root['threshold']:
                left.append(row)
            else:
                right.append(row)
        groups_subtree[0] = left
        groups_subtree[1] = right
    elif (root['type'] == 'D'):
        for row in data:
            for attribute_i in range(len(root['groups'])):
                if row[index] == root['groups'][attribute_i]:
                    groups_subtree[attribute_i].append(row)
    return groups_subtree


# Create a terminal node value
def to_terminal(data):
    outcomes = [row[-1] for row in data]
    return max(set(outcomes), key=outcomes.count)


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def DecisionTreeBounded(max_depth):
    #      This function should return a list of labels.
    #      e.g.:
    #	labels = [['+','-','+'],['+','+','-'],['-','+'],['+','+']]
    #	return labels
    #	where:
    #		labels[0] = original_training_labels
    #		labels[1] = prediected_training_labels
    #		labels[2] = original_testing_labels
    #		labels[3] = predicted_testing_labels

    import Check as C
    labels = list(list())

    # Build DT from training set
    training_rows = C.readfile("train.txt")

    # Initializations
    attribute_list = list(range(0, 15))
    root = {}
    current_depth = 0
    attribute_info = configure_attribute_info(attribute_list)

    # impute_missing_data(rows)
    impute_missing_data(training_rows, attribute_info)

    # Train DT
    root = build_tree(training_rows, root, attribute_list, current_depth, max_depth, attribute_info)

    # Save original labels in labels[0]
    labels.append(get_dataset_class_labels(training_rows))
    # Predict training set and save in labels[1]
    labels.append(predict_dataset(training_rows, root))
    # Load Testing Set - labels in labels[2]
    testing_rows = C.readfile("test.txt")
    impute_missing_data(testing_rows, attribute_info)
    labels.append(get_dataset_class_labels(testing_rows))
    # Predict Testing Set - labels[3]
    labels.append(predict_dataset(testing_rows, root))

    return labels
