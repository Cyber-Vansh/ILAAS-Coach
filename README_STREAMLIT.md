# 🎓 ILA Coach Dashboard

A modern, interactive Streamlit dashboard for Intelligent Learning Analytics Coach. This application provides comprehensive student risk analysis, data visualization, and predictive analytics for educational institutions.

## ✨ Features

### 📊 **Dashboard**
- Real-time student metrics and KPIs
- Interactive charts and visualizations
- Risk level distribution analysis
- Recent alerts and notifications
- GPA and attendance trends

### 📈 **Data Visualization**
- Advanced filtering capabilities
- Correlation heatmaps
- 3D scatter plots
- Parallel coordinates analysis
- Box plots and distribution charts
- Multi-dimensional data exploration

### 🎯 **Risk Analysis**
- Individual student risk assessment
- ML-powered predictive analytics
- Feature importance analysis
- Personalized recommendations
- Real-time risk scoring

### ⚙️ **Settings**
- Customizable display options
- Notification preferences
- Privacy and data retention settings
- System information

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /Users/sohamsaranga/Desktop/ila-coach/ILAAS-Coach
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser:**
   Navigate to `http://localhost:8501`

## 🎨 UI Features

### Modern Design
- **Gradient backgrounds** and smooth animations
- **Responsive layout** that works on all devices
- **Color-coded risk indicators** (Green: Low, Orange: Medium, Red: High)
- **Interactive hover effects** and transitions
- **Professional typography** and spacing

### Interactive Components
- **Real-time data updates** with caching
- **Dynamic filtering** by program, year, and risk level
- **Interactive charts** using Plotly
- **Expandable sections** and collapsible content
- **Custom CSS styling** for enhanced visuals

### Data Visualizations
- **Pie charts** for risk distribution
- **Scatter plots** for correlation analysis
- **3D visualizations** for multi-dimensional data
- **Heatmaps** for correlation matrices
- **Bar charts** for categorical data

## 📊 Data Structure

The application works with student data including:

- **Academic Metrics:** GPA, attendance rate, assignment completion
- **Behavioral Data:** Study hours, extracurricular activities
- **Demographics:** Age, program, year of study
- **Risk Assessment:** Automated risk scoring and classification
- **Temporal Data:** Last login, activity timestamps

## 🔧 Configuration

### Model Integration
The app automatically integrates with your existing ML models:
- `src/student_risk_model.pkl` - Trained risk prediction model
- `src/scaler.pkl` - Data preprocessing scaler

### Customization Options
- **Theme selection:** Light, Dark, or Auto
- **Color schemes:** Purple, Blue, Green, Orange
- **Refresh intervals:** Configurable data update frequency
- **Alert thresholds:** Customizable risk level triggers

## 🎯 Key Features Explained

### Risk Analysis System
- **Automated scoring** based on multiple factors
- **Three-tier classification:** Low, Medium, High risk
- **Predictive modeling** using machine learning
- **Actionable recommendations** for each risk level

### Interactive Dashboard
- **Real-time metrics** with trend indicators
- **Quick stats** in the sidebar
- **Recent alerts** for high-risk students
- **Performance visualizations**

### Advanced Analytics
- **Correlation analysis** between different metrics
- **Multi-dimensional exploration** with 3D plots
- **Feature importance** visualization
- **Comparative analysis** across programs

## 🛠️ Technical Stack

- **Frontend:** Streamlit 1.54.0
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Styling:** Custom CSS with gradients and animations

## 📱 Responsive Design

The dashboard is fully responsive and works seamlessly on:
- **Desktop computers** (full feature set)
- **Tablets** (optimized layout)
- **Mobile devices** (compact view)

## 🔒 Privacy & Security

- **Data anonymization** options
- **Configurable retention periods**
- **Access logging** capabilities
- **Secure model integration**

## 🚀 Performance Features

- **Intelligent caching** for fast loading
- **Lazy loading** of large datasets
- **Optimized rendering** for smooth interactions
- **Background data refresh**

## 🎨 Visual Design Elements

### Color Scheme
- **Primary:** Purple gradient (#667eea → #764ba2)
- **Success:** Green (#26de81)
- **Warning:** Orange (#ffa502)
- **Danger:** Red (#ff4757)

### Typography
- **Headers:** Bold, gradient text
- **Metrics:** Large, clear numbers
- **Body:** Clean, readable fonts

### Animations
- **Hover effects** on buttons and cards
- **Smooth transitions** between sections
- **Loading animations** for data updates

## 📞 Support

For issues or questions:
1. Check the console for error messages
2. Verify all dependencies are installed
3. Ensure model files are present in `src/` directory
4. Check data file permissions

## 🔄 Updates

The application supports:
- **Live data refresh** without page reload
- **Configuration persistence**
- **Model hot-swapping**
- **Dynamic content updates**

---

**© 2024 ILA Coach - Intelligent Learning Analytics Dashboard**

Built with ❤️ using Streamlit and modern web technologies.
