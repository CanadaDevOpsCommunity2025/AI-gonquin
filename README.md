# DevOps for GenAI Hackathon Chatbot

A modern, interactive chatbot application that combines DevOps practices with Generative AI capabilities. Built with FastAPI, React, and WebSocket for real-time communication, this chatbot provides insights into DevOps processes, GenAI integration, and real-time monitoring.

## 🌟 Features

### Core Features
- Real-time chat interface with WebSocket communication
- DevOps process monitoring and automation
- GenAI-powered insights and analysis
- Dark-themed, modern UI with smooth animations
- Responsive design for all devices

### DevOps Integration
- CI/CD pipeline monitoring and automation
- Infrastructure as Code (IaC) management
- Container orchestration and management
- Cloud resource monitoring and optimization
- Security and compliance tracking
- Performance metrics and analytics

### GenAI Capabilities
- AI-powered code analysis and suggestions
- Automated testing and quality assurance
- Intelligent deployment strategies
- Predictive maintenance and monitoring
- Natural language processing for logs
- Automated documentation generation

### Voice Integration
- Speech-to-text for voice commands
- Text-to-speech for system status
- Multi-language support (English, Spanish, French, German)
- Voice feedback for all interactions
- Hands-free operation

### Monitoring & Analytics
- Real-time system metrics
- Performance analytics
- Resource utilization tracking
- Security monitoring
- Cost optimization insights
- Trend analysis and predictions

## 🛠 Tech Stack

### Frontend
- React with TypeScript
- Material-UI for components
- WebSocket for real-time communication
- Web Speech API for voice features
- Styled Components for styling

### Backend
- FastAPI (Python)
- WebSocket support
- GenAI model integration
- DevOps tool integration
- Multi-language support

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Modern web browser with Web Speech API support
- Access to DevOps tools and services

### Installation

1. Clone the repository:
```bash
git clone https://github.com/dalalelamine/Devops-chatbot.git
cd Devops-chatbot
```

2. Create a `.env` file in the root directory:
```env
NEWS_API_KEY=your_news_api_key
REACT_APP_API_URL=http://localhost:8000
POSTGRES_PASSWORD=your_postgres_password
```

3. Start the application:
```bash
docker compose up --build
```

4. Access the application at `http://localhost:8080`

## 💬 Usage Guide

### DevOps Commands
```
/devops - Show DevOps-related commands
/ci - Show CI/CD pipeline status
/deploy - Show deployment status
/monitor - Show system monitoring metrics
/logs - Show recent system logs
/containers - Show running containers
/kubernetes - Show Kubernetes cluster status
/terraform - Show Terraform state
/ansible - Show Ansible playbook status
/jenkins - Show Jenkins pipeline status
/gitlab - Show GitLab CI/CD status
/github - Show GitHub Actions status
/docker - Show Docker status
/aws - Show AWS resources
/azure - Show Azure resources
/gcp - Show GCP resources
/security - Show security scan results
/backup - Show backup status
/restore - Show restore points
/performance - Show performance metrics
```

### GenAI DevOps Commands
```
/hackathon - Show DevOps for GenAI Hackathon news
/genai - Show GenAI and DevOps integration news
/mlops - Show MLOps and DevOps news
/aiops - Show AIOps and DevOps news
/automation - Show AI automation in DevOps news
/monitoring - Show AI-powered monitoring news
/security - Show AI in DevOps security news
/testing - Show AI in testing and QA news
/deployment - Show AI in deployment automation news
/integration - Show AI-DevOps integration news
```

### Voice Commands
```
/voice - Toggle voice mode
/read - Read status aloud
/preferences - Set preferences
/feedback - Provide feedback
```

## 🔧 Configuration

### Environment Variables
- `NEWS_API_KEY`: Your News API key
- `REACT_APP_API_URL`: Backend API URL
- `DEVOPS_TOOLS`: Configured DevOps tools
- `GENAI_MODEL`: Selected GenAI model
- `POSTGRES_PASSWORD`: PostgreSQL password

### User Preferences
Customize your experience through:
- DevOps tool selection
- Monitoring preferences
- Language settings
- Voice settings
- Notification preferences

## 🐛 Troubleshooting

### Common Issues
1. **Voice Recognition Not Working**
   - Check microphone permissions
   - Ensure browser supports Web Speech API
   - Try refreshing the page

2. **DevOps Tools Not Responding**
   - Verify tool configurations
   - Check API credentials
   - Ensure network connectivity

3. **GenAI Model Issues**
   - Check model availability
   - Verify API keys
   - Monitor resource usage

### Browser Support
- Chrome (recommended)
- Firefox
- Edge
- Safari (limited voice support)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [News API](https://newsapi.org) for news data
- [Material-UI](https://mui.com) for UI components
- [FastAPI](https://fastapi.tiangolo.com) for backend framework
- [React](https://reactjs.org) for frontend framework

## 📞 Support

For support, please:
1. Check the [troubleshooting guide](#troubleshooting)
2. Open an issue in the repository
3. Contact the maintainers

## 🔄 Updates

### Latest Updates
- Added GenAI integration
- Enhanced DevOps monitoring
- Improved voice capabilities
- Added multi-language support
- Enhanced security features

### Planned Features
- Advanced GenAI models
- Extended DevOps tool support
- Enhanced monitoring capabilities
- User authentication
- Team collaboration features 