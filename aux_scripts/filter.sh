echo "Filtering data..."
python3 aux_scripts/data_filtering.py > salida_filtering.txt

echo "Getting reference corpus data..."
python3 aux_scripts/get_reference_corpus_data.py