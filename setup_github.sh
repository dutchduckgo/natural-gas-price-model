#!/bin/bash

# Natural Gas Price Model - GitHub Repository Setup Script
# This script helps you create a GitHub repository for your project

echo "🚀 Setting up GitHub repository for Natural Gas Price Model"
echo "=========================================================="

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository. Please run 'git init' first."
    exit 1
fi

# Check if we have commits
if [ -z "$(git log --oneline 2>/dev/null)" ]; then
    echo "❌ No commits found. Please commit your changes first."
    exit 1
fi

echo "✅ Git repository initialized with commits"

# Check if GitHub CLI is installed
if command -v gh &> /dev/null; then
    echo "✅ GitHub CLI found"
    
    # Check if user is authenticated
    if gh auth status &> /dev/null; then
        echo "✅ GitHub CLI authenticated"
        
        # Create repository
        echo "📦 Creating GitHub repository..."
        gh repo create natural-gas-price-model \
            --public \
            --description "Comprehensive U.S. LNG/natural gas price prediction model with multi-level ML architecture, data ingestion pipeline, and production-ready features" \
            --add-readme \
            --clone=false
            
        if [ $? -eq 0 ]; then
            echo "✅ Repository created successfully!"
            
            # Add remote origin
            echo "🔗 Adding remote origin..."
            git remote add origin https://github.com/$(gh api user --jq .login)/natural-gas-price-model.git
            
            # Push to GitHub
            echo "📤 Pushing code to GitHub..."
            git branch -M main
            git push -u origin main
            
            if [ $? -eq 0 ]; then
                echo "🎉 Success! Your repository is now available at:"
                echo "   https://github.com/$(gh api user --jq .login)/natural-gas-price-model"
            else
                echo "❌ Failed to push to GitHub"
                exit 1
            fi
        else
            echo "❌ Failed to create repository"
            exit 1
        fi
    else
        echo "⚠️  GitHub CLI not authenticated. Please run:"
        echo "   gh auth login"
        echo "   Then run this script again."
        exit 1
    fi
else
    echo "⚠️  GitHub CLI not found. Please follow these manual steps:"
    echo ""
    echo "1. Go to https://github.com/new"
    echo "2. Create a new repository with these settings:"
    echo "   - Repository name: natural-gas-price-model"
    echo "   - Description: Comprehensive U.S. LNG/natural gas price prediction model with multi-level ML architecture, data ingestion pipeline, and production-ready features"
    echo "   - Visibility: Public"
    echo "   - Initialize with README: Yes"
    echo ""
    echo "3. After creating the repository, run these commands:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/natural-gas-price-model.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
    echo "4. Your repository will be available at:"
    echo "   https://github.com/YOUR_USERNAME/natural-gas-price-model"
fi

echo ""
echo "📋 Repository Features:"
echo "   ✅ Complete data ingestion pipeline (EIA, weather, power)"
echo "   ✅ Multi-level ML models (baseline, tree-based, deep learning)"
echo "   ✅ Feature engineering for weather, storage, and market data"
echo "   ✅ Backtesting and evaluation framework"
echo "   ✅ Production-ready pipeline with comprehensive documentation"
echo "   ✅ Jupyter notebook examples and demo script"
echo ""
echo "🎯 Next Steps:"
echo "   1. Install dependencies: pip install -r requirements.txt"
echo "   2. Run demo: python demo.py"
echo "   3. Explore examples: jupyter notebook notebooks/example_usage.ipynb"
echo "   4. Read documentation: README.md and IMPLEMENTATION_SUMMARY.md"
