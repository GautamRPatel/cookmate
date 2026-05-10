1) Create a virtual environment using following command
    python -m venv YOUR_ENV_NAME

2) Run the environment
    YOUR_ENV_NAME\scripts\activate

3) Install required dependecies
    pip install -r requirements.txt

4) Run follwing command to test in same machine(laptop)
    uvicorn main:app --reload --port 5000
    
5) Run this command if want to test from any device in the same network(Emulator,mobile,etc)
    uvicorn main:app --host 0.0.0.0 --port 8000