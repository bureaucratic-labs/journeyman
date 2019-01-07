# journeyman
Character-level bidirectional recurrent neural network for short sentences categorization

# Install

```bash
$ pip install journeyman-nn
```

# Usage

Train data format (text and label splitted by `\t`):

```
двухкомнатной квартире  2
трехкомнатной квартире  3
3-кк    3
1 . 1
5 комнат    5
3 квартире  3
3-к квартире    3
```

Training:

```python
from journeyman import Sequence, load_data_and_labels

train_x, train_y = load_data_and_labels('train.txt')

model = Sequence(dropout=0.5, embedding_dim=256, units=128, maxlen=32, batch_size=32)
model.fit(train_x, train_y, epochs=20)
model.save('model.h5', 'preprocessor.pickle', 'params.json')
```

Usage:

```python
from journeyman import load_model

model = load_model('model.h5', 'preprocessor.pickle', 'params.json')
model.predict(['1 к кв.', 'двушка', 'трёхкомнатная квартира'])  # => ['1' '2' '3']
```
