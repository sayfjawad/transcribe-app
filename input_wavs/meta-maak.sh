# Zorg voor een lege transcript.txt
> transcript.txt

# Voeg alle teksten samen in alfabetische volgorde van wav-bestanden
for i in *.txt; do
  txtfile="${i%}"
  cat "$txtfile" >> transcript.txt
  echo "" >> transcript.txt
done
