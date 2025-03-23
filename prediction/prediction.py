import argparse
import os
import numpy as np
import torch
from Bio import SeqIO
from Bio.Align import PairwiseAligner, substitution_matrices
from models.CNN import CNN
from models.GloBCEpred import GloBCEpred

# Define the amino acid encoding dictionary
amino_acid_encoding = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7,
    "K": 8, "L": 9, "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14, "S": 15,
    "T": 16, "V": 17, "W": 18, "Y": 19, "X": 20
}

# One-hot encoding
def one_hot_encode(sequence, encoding_dict):
    encoding = np.zeros((len(sequence), len(encoding_dict)))
    for i, amino_acid in enumerate(sequence):
        if amino_acid in encoding_dict:
            encoding[i, encoding_dict[amino_acid]] = 1
        else:
            encoding[i, encoding_dict["X"]] = 1
    return encoding

# Positional encoding
class PositionalEncoding:
    def __init__(self, d_model):
        self.d_model = d_model

    def get_encoding(self, position):
        encoding = torch.zeros(self.d_model)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        encoding[0::2] = torch.sin(position * div_term)
        encoding[1::2] = torch.cos(position * div_term)
        return encoding

# Load set2 sequences
def load_set2_sequences(set2_path):
    target_ids = {
    "A0A044SS43", "A0A044U9P0", "A0A044UK00", "A0A044UVG8", "A0A059VFK8", "A0A060N479",
    "A0A0H5BN46", "A0A0P0ILY9", "A0A0P0K6L9", "A0A0P0KXN3", "A0A0P0LKV0", "A0A0U1RQC9",
    "A0A140H1G9", "A0A140H1H1", "A0A378NCQ9", "A0A3B3IU09", "A0A3Q0KF32", "A0A4E0RUD3",
    "A0A4Z2D725", "A0A4Z2DAE1", "A0A669KB41", "A0A7I2V599", "B4YAH6", "B9TD97", "D2T2K3",
    "F5HB53", "G4VHH4", "H9EJ24", "H9EJ63", "H9EJ69", "J3QL64", "K9N4V0", "K9N4V7",
    "K9N5Q8", "K9N638", "K9N7C7", "K9N7D2", "O09710", "O77463", "O82580", "P00451",
    "P00734", "P01005", "P01012", "P01024", "P01031", "P01266", "P02458", "P02538",
    "P02666", "P02668", "P02675", "P02708", "P02730", "P02749", "P02751", "P03101",
    "P03129", "P03198", "P03200", "P03203", "P03204", "P03300", "P03378", "P03381",
    "P04013", "P04114", "P04739", "P04925", "P04958", "P05067", "P05783", "P06733",
    "P06821", "P07131", "P08238", "P08363", "P08603", "P09012", "P0A3R5", "P0C0L4",
    "P0C6U8", "P0C6X1", "P0C6X6", "P0C6X7", "P0C767", "P0CE82", "P0DH58", "P0DTC2",
    "P0DTC3", "P0DTC5", "P0DTC8", "P0DTD1", "P10155", "P10279", "P10636", "P13285",
    "P13290", "P13423", "P14283", "P15130", "P15423", "P15494", "P15941", "P16055",
    "P16473", "P20930", "P21980", "P22303", "P23093", "P23907", "P24158", "P27313",
    "P27455", "P29723", "P35418", "P43238", "P43838", "P49450", "P59594", "P59595",
    "P59596", "P59636", "P60202", "P62314", "P9WPE7", "P9WQP1", "Q01955", "Q02224",
    "Q0ZJ80", "Q0ZJ82", "Q0ZME3", "Q13018", "Q14624", "Q14CZ8", "Q155Z9", "Q2FXX5",
    "Q32ZE1", "Q3LZX1", "Q3LZX4", "Q4DX15", "Q4QXJ7", "Q4VID0", "Q647G8", "Q69091",
    "Q6SW59", "Q73WG6", "Q76R62", "Q77YG2", "Q786F2", "Q7K740", "Q80872", "Q83884",
    "Q8I0U6", "Q8I2B4", "Q8I639", "Q8IBE8", "Q8IJ55", "Q8ILZ1", "Q95PM1", "Q99XV0",
    "Q9BYF1", "Q9DLK1", "Q9PZT0", "Q9Q6P4", "Q9UM07", "Q9UMD9", "U5WLB9", "X6RKS3"}
    sequences = []
    for record in SeqIO.parse(set2_path, "fasta"):
        if record.id in target_ids:
            sequences.append(str(record.seq))
    return sequences

# Calculate scores
def calculate_scores(query_sequence, all_sequences):
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = torch.zeros(len(all_sequences), device=device)

    for j, seq2 in enumerate(all_sequences):
        score = aligner.score(query_sequence, seq2)
        scores[j] = score

    return scores.cpu().numpy()

# Generate data by sliding window
def generate_windowed_data(sequence, window_size, coordinates_data):
    X = []
    pe = PositionalEncoding(10)
    padding_length = window_size // 2

    encoded_padding = np.zeros((padding_length, len(amino_acid_encoding)))
    encoded_sequence = one_hot_encode(sequence, amino_acid_encoding)
    padded_sequence = np.vstack([encoded_padding, encoded_sequence, encoded_padding])

    for i in range(len(sequence)):
        window_seq = padded_sequence[i:i + window_size]
        start_index = i + 1
        end_index = i + window_size

        start_pos_encoding = pe.get_encoding(start_index).numpy()
        end_pos_encoding = pe.get_encoding(end_index).numpy()

        coord_data = np.tile(coordinates_data, (window_size, 1))
        start_pos_encoding = np.tile(start_pos_encoding, (window_size, 1))
        end_pos_encoding = np.tile(end_pos_encoding, (window_size, 1))

        window_features = np.concatenate(
            [window_seq, coord_data, start_pos_encoding, end_pos_encoding], axis=1
        )
        X.append(window_features)

    return np.array(X)

# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input FASTA file')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    args = parser.parse_args()

    # Create the output directory
    os.makedirs(args.output, exist_ok=True)

    # Load set2 sequences
    set2_sequences = load_set2_sequences('../data/train_data/train.fasta')

    # Initialize the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CNN model
    cnn = CNN().to(device)
    cnn.load_state_dict(torch.load('../models/cnn.pth', map_location=device))
    cnn.eval()

    # Load GloBCEpred model
    gbepred = GloBCEpred().to(device)
    gbepred.load_state_dict(torch.load('../models/parameter.pth', map_location=device))
    gbepred.eval()

    # Process input sequences
    for record in SeqIO.parse(args.input, "fasta"):
        seq_id = record.id
        sequence = str(record.seq)

        # calculate scores
        scores = calculate_scores(sequence, set2_sequences)

        # Extract coordinates using the CNN model
        with torch.no_grad():
            scores_tensor = torch.tensor(scores).float().to(device)
            coordinates = cnn(scores_tensor.unsqueeze(0)).cpu().numpy()

        # Generate data with sliding windows
        window_size = 11
        X = generate_windowed_data(sequence, window_size, coordinates)
        X_tensor = torch.tensor(X).float().to(device)

        # Perform prediction
        with torch.no_grad():
            predictions = gbepred(X_tensor).cpu().numpy().flatten()

        # Save results
        output_path = os.path.join(args.output, f"{seq_id}.csv")
        positions = np.arange(1, len(sequence) + 1)
        np.savetxt(output_path,
                   np.column_stack((positions, predictions)),
                   delimiter=',',
                   header='position,predicted_response_frequency',
                   comments='',
                   fmt=['%d', '%.4f'])

if __name__ == "__main__":
    main()
