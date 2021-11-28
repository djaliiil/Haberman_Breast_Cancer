// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';

// Pie Chart Status
var ctx_status = document.getElementById("myPieChartStatus");
var myPieChartStatus = new Chart(ctx_status, {
  type: 'doughnut',
  data: {
    labels: ["Survive", "Dead"],
    datasets: [{
      data: [225, 81],
      backgroundColor: ['#4e73df', '#1cc88a'],
      hoverBackgroundColor: ['#2e59d9', '#17a673'],
      hoverBorderColor: "rgba(234, 236, 244, 1)",
    }],
  },
  options: {
    maintainAspectRatio: false,
    tooltips: {
      backgroundColor: "rgb(255,255,255)",
      bodyFontColor: "#858796",
      borderColor: '#dddfeb',
      borderWidth: 1,
      xPadding: 15,
      yPadding: 15,
      displayColors: true,
      caretPadding: 10,
    },
    legend: {
      display: false
    },
    cutoutPercentage: 80,
  },
});


// Pie Chart Node
var ctx_node = document.getElementById("myPieChartNode");
var myPieChartNode = new Chart(ctx_node, {
  type: 'doughnut',
  data: {
    labels: ['0-Nodes','1-Nodes','2-Nodes','3-Nodes','4-Nodes','8-Nodes','7-Nodes','6-Nodes','9-Nodes','5-Nodes','13-Nodes','11-Nodes','14-Nodes','23-Nodes','15-Nodes','22-Nodes','10-Nodes','19-Nodes','12-Nodes','20-Nodes','25-Nodes','18-Nodes','46-Nodes','28-Nodes','35-Nodes','17-Nodes','52-Nodes','21-Nodes','16-Nodes','24-Nodes','30-Nodes'],
    datasets: [{
      data: [136,41,20,20,13,7,7,7,6,6,5,4,4,3,3,3,3,3,2,2,1,1,1,1,1,1,1,1,1,1,1],
      backgroundColor: ['#4e73df', '#1cc88a', '#4e73df', '#1cc88a', '#4e73df', '#1cc88a','#4e73df', '#1cc88a','#4e73df', '#1cc88a','#4e73df', '#1cc88a','#4e73df', '#1cc88a','#4e73df', '#1cc88a','#4e73df', '#1cc88a','#4e73df', '#1cc88a','#4e73df', '#1cc88a','#4e73df', '#1cc88a','#4e73df', '#1cc88a','#4e73df', '#1cc88a','#4e73df', '#1cc88a','#4e73df'],
      hoverBackgroundColor: ['#2e59d9', '#17a673', '#2e59d9', '#17a673', '#2e59d9', '#17a673', '#2e59d9', '#17a673', '#2e59d9', '#17a673', '#2e59d9', '#17a673', '#2e59d9', '#17a673','#2e59d9', '#17a673', '#2e59d9', '#17a673', '#2e59d9', '#17a673', '#2e59d9', '#17a673', '#2e59d9', '#17a673', '#2e59d9', '#17a673', '#2e59d9', '#17a673', '#2e59d9'],
      hoverBorderColor: "rgba(234, 236, 244, 1)",
    }],
  },
  options: {
    maintainAspectRatio: false,
    tooltips: {
      backgroundColor: "rgb(255,255,255)",
      bodyFontColor: "#858796",
      borderColor: '#dddfeb',
      borderWidth: 1,
      xPadding: 15,
      yPadding: 15,
      displayColors: true,
      caretPadding: 10,
    },
    legend: {
      display: false
    },
    cutoutPercentage: 80,
  },
});
