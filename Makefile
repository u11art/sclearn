on:
	source .venv/bin/activate

train:
	python model.py

start:
	python app.py

update:
	pip install -r requirements.txt