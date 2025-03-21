# GloBCEpred

``GloBECpred`` is a deep learning framework for B-cell epitope prediction. It is trained on experimentally derived response frequency data and integrates both local sequence patterns and global antigen-specific features.

# Installation

1.  Clone the repository
```
git clone https://github.com/ZihanWang1029/GloBECpred
```
2.   (Recommended) Create a virtual environment
```conda deactivate
conda create --name GloBECpred python=3.11
conda activate GloBECpred
```
3.  Install required dependencies
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pandas numpy Bio
```

#  Usage

Run the prediction script with your input FASTA file. Example:
```
cd GloBCEpred/prediction/
python prediction.py -f ../testdata/test.fasta -o ../result/test_result
```
Arguments:
-   `-f`: Path to the input FASTA file.
-   `-o`: Path to the output result directory.

# Reference
Zihan, W. et al. (2025+), "GloBECpred: Incorporating Global and Local Information for B-cell Epitope Curve Prediction", working paper.
