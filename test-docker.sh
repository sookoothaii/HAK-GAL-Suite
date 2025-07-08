#!/bin/bash

# Test script for HAK-GAL Suite Docker setup

echo "🐳 Testing HAK-GAL Suite Docker Setup"
echo "======================================"

# Test backend API
echo "1. Testing Backend API:"
echo "   Endpoint: http://localhost:5001/api/test"
echo "   Response:"
curl -s http://localhost:5001/api/test | python -m json.tool
echo ""

# Test frontend accessibility
echo "2. Testing Frontend:"
echo "   Endpoint: http://localhost:5173"
echo "   Status:"
curl -s -I http://localhost:5173 | head -n 1
echo ""

# Test CORS by making a request from frontend port to backend
echo "3. Testing CORS Configuration:"
echo "   Testing if frontend can reach backend..."
curl -s -X OPTIONS http://localhost:5001/api/command \
  -H "Origin: http://localhost:5173" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type" \
  -I | head -n 3
echo ""

echo "✅ Tests completed!"