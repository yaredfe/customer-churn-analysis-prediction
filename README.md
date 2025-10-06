# Customer Churn Analysis & Prediction

A comprehensive machine learning solution for predicting customer churn in telecommunications companies, built with modern DevOps practices and cloud infrastructure.

## 🎯 Project Overview

This project demonstrates a complete MLOps pipeline for customer churn prediction, featuring:

- **Machine Learning Pipeline**: Data preprocessing, feature engineering, model training, and evaluation
- **API Service**: FastAPI backend for model serving and predictions
- **Web Interface**: Flask frontend for user interaction
- **Containerization**: Docker-based deployment with load balancing
- **Infrastructure as Code**: Terraform for AWS cloud provisioning
- **CI/CD Pipeline**: Automated deployment with GitHub Actions
- **Configuration Management**: Ansible for server configuration

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Flask Web     │    │   Nginx LB      │    │   FastAPI       │
│   Frontend      │◄──►│   (Reverse      │◄──►│   Backend       │
│   (Port 5000)   │    │    Proxy)       │    │   (Port 8000)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Compose                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │ Frontend-1  │ │ Frontend-2  │ │   API-1     │ │   API-2     ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
│  ┌─────────────┐                                               │
│  │   API-3     │                                               │
│  └─────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AWS Infrastructure                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   EC2 Instance                              ││
│  │  ┌─────────────────────────────────────────────────────────┐││
│  │  │              Application Stack                          │││
│  │  │  • Docker Containers                                   │││
│  │  │  • Nginx Load Balancer                                  │││
│  │  │  • ML Models (XGBoost, LightGBM)                       │││
│  │  │  • FastAPI + Flask Services                            │││
│  │  └─────────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Features

### Machine Learning
- **Data Analysis**: Comprehensive EDA with visualizations
- **Feature Engineering**: Automated preprocessing pipeline
- **Model Training**: Multiple algorithms (XGBoost, LightGBM, Random Forest)
- **Model Evaluation**: Cross-validation and performance metrics
- **Model Serving**: RESTful API for predictions

### Infrastructure
- **Load Balancing**: Round-robin distribution across multiple instances
- **Auto-scaling**: Container-based horizontal scaling
- **High Availability**: Multi-instance deployment
- **Reverse Proxy**: Nginx for request routing and SSL termination

### DevOps
- **Infrastructure as Code**: Terraform for AWS provisioning
- **Container Orchestration**: Docker Compose for local development
- **Configuration Management**: Ansible for server setup
- **CI/CD Pipeline**: Automated testing and deployment
- **Monitoring**: Application health checks and logging

## 🛠️ Tech Stack

### Backend & ML
- **Python 3.11**: Core programming language
- **FastAPI**: Modern, fast web framework for APIs
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Microsoft's gradient boosting framework
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning library
- **Joblib**: Model serialization

### Frontend
- **Flask**: Lightweight web framework
- **HTML/CSS/JavaScript**: Frontend technologies
- **Bootstrap**: CSS framework for responsive design

### Infrastructure & DevOps
- **Docker**: Containerization platform
- **Docker Compose**: Multi-container orchestration
- **Nginx**: Web server and reverse proxy
- **Terraform**: Infrastructure as Code
- **Ansible**: Configuration management
- **GitHub Actions**: CI/CD pipeline
- **AWS**: Cloud infrastructure (EC2, VPC, Security Groups)

## 📁 Project Structure

```
customer-churn-analysis-prediction/
├── 📁 src/                          # Source code
│   ├── 📁 api/                      # FastAPI backend
│   ├── 📁 config/                   # Configuration files
│   ├── 📁 data/                     # Data handling modules
│   ├── 📁 eda/                      # Exploratory data analysis
│   ├── 📁 frontend_flask/           # Flask frontend
│   ├── 📁 models/                   # ML model definitions
│   ├── 📁 pipeline/                 # Data preprocessing pipeline
│   ├── 📁 utils/                    # Utility functions
│   └── 📁 notebooks/                # Jupyter notebooks
├── 📁 artifacts/                    # Model artifacts
│   └── 📁 models/                   # Trained models
├── 📁 scripts/                      # Utility scripts
├── 📁 terraform/                    # Infrastructure as Code
├── 📁 ansible/                      # Configuration management
├── 📁 deploy/                       # Deployment configurations
├── 📁 reports/                      # Analysis reports
├── 📁 .github/workflows/            # CI/CD pipelines
├── 📄 docker-compose.yml            # Container orchestration
├── 📄 Dockerfile.api                # API container
├── 📄 Dockerfile.frontend           # Frontend container
├── 📄 requirements.txt              # Python dependencies
└── 📄 README.md                     # Project documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Git

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/customer-churn-analysis-prediction.git
   cd customer-churn-analysis-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run with Docker Compose (Recommended)**
   ```bash
   docker-compose up --build
   ```

4. **Access the application**
   - Web Interface: http://localhost
   - API Documentation: http://localhost/api/docs
   - Health Check: http://localhost/api/health

### Manual Setup

1. **Start the API server**
   ```bash
   python scripts/serve.py
   ```

2. **Start the frontend (in another terminal)**
   ```bash
   python scripts/frontend.py
   ```

## 🌐 API Documentation

### Endpoints

- `GET /api/health` - Health check
- `GET /api/docs` - Interactive API documentation
- `POST /api/predict` - Make predictions
- `GET /api/model/info` - Model information

### Example Prediction Request

```bash
curl -X POST "http://localhost/api/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "customer_data": {
         "tenure": 12,
         "monthly_charges": 70.5,
         "total_charges": 846.0,
         "contract": "Month-to-month",
         "internet_service": "DSL"
       }
     }'
```

## ☁️ Cloud Deployment

### AWS Infrastructure

1. **Set up AWS credentials**
   ```bash
   export AWS_ACCESS_KEY_ID="your-access-key"
   export AWS_SECRET_ACCESS_KEY="your-secret-key"
   ```

2. **Deploy infrastructure**
   ```bash
   cd terraform
   terraform init
   terraform plan
   terraform apply
   ```

3. **Configure Ansible hosts**
   ```bash
   cp ansible/hosts.example ansible/hosts
   # Update with your EC2 public IP
   ```

4. **Deploy application**
   ```bash
   ansible-playbook -i ansible/hosts ansible/playbook.yml
   ```

### GitHub Actions CI/CD

The project includes automated deployment via GitHub Actions:

1. **Set up GitHub Secrets**:
   - `DOCKER_USERNAME`: Docker Hub username
   - `DOCKER_TOKEN`: Docker Hub access token
   - `EC2_HOST`: EC2 public IP address
   - `EC2_USER`: EC2 username (ubuntu)
   - `EC2_SSH_KEY`: SSH private key

2. **Push to main branch** triggers automatic deployment

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 0.847 | 0.812 | 0.789 | 0.800 |
| LightGBM | 0.843 | 0.808 | 0.785 | 0.796 |
| Random Forest | 0.835 | 0.801 | 0.778 | 0.789 |

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
API_URL=http://localhost:8000
PYTHONUNBUFFERED=1
```

### Terraform Variables

Update `terraform/terraform.tfvars`:

```hcl
region = "us-east-1"
instance_name = "customer-churn-app"
instance_type = "t3.micro"
key_name = "your-ec2-key"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Yared Fereja**
- GitHub: [@yaredfe](https://github.com/yaredfe)

## 🙏 Acknowledgments

- Telco Customer Churn Dataset
- AWS for cloud infrastructure
- Open source community for tools and libraries

---

**Built with ❤️ for Infrastructure Engineering Excellence**