#!/bin/bash

cd "$(dirname "$0")/.."

case "$1" in
    api)
        echo "Starting API server..."
        uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    dashboard)
        echo "Starting dashboard..."
        cd dashboard && npm run dev
        ;;
    demo)
        echo "Running demo..."
        python scripts/demo.py
        ;;
    install)
        echo "Installing dependencies..."
        pip install -r requirements.txt
        cd dashboard && npm install
        ;;
    blockchain)
        echo "Starting local blockchain..."
        cd blockchain && npx hardhat node
        ;;
    deploy)
        echo "Deploying contracts..."
        cd blockchain && npx hardhat run scripts/deploy.js --network localhost
        ;;
    all)
        echo "Starting all services..."
        trap 'kill $(jobs -p)' EXIT
        uvicorn api.main:app --host 0.0.0.0 --port 8000 &
        cd dashboard && npm run dev &
        wait
        ;;
    *)
        echo "SENTINEL - Botnet Detection System"
        echo ""
        echo "Usage: $0 {api|dashboard|demo|install|blockchain|deploy|all}"
        echo ""
        echo "Commands:"
        echo "  api        - Start the FastAPI backend"
        echo "  dashboard  - Start the React dashboard"
        echo "  demo       - Run the demo script"
        echo "  install    - Install all dependencies"
        echo "  blockchain - Start local Hardhat node"
        echo "  deploy     - Deploy smart contracts"
        echo "  all        - Start API and dashboard together"
        ;;
esac

