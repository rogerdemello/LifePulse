// Enhanced Result Card Components for LifePulse

// ========================================
// ANIMATED PROGRESS CIRCLE
// ========================================
function createProgressCircle(containerId, score, color = '#667eea', label = 'Score') {
  const container = document.getElementById(containerId);
  if (!container) return;

  const size = 220;
  const strokeWidth = 18;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = (score / 100) * circumference;

  container.innerHTML = `
    <div class="progress-circle-wrapper">
      <svg width="${size}" height="${size}" class="progress-circle-svg">
        <!-- Background circle -->
        <circle 
          cx="${size/2}" 
          cy="${size/2}" 
          r="${radius}" 
          fill="none" 
          stroke="#e0e0e0" 
          stroke-width="${strokeWidth}"
        />
        <!-- Progress circle -->
        <circle 
          class="progress-circle-fill"
          cx="${size/2}" 
          cy="${size/2}" 
          r="${radius}" 
          fill="none" 
          stroke="${color}" 
          stroke-width="${strokeWidth}"
          stroke-dasharray="0 ${circumference}"
          stroke-linecap="round"
          transform="rotate(-90 ${size/2} ${size/2})"
        />
        <!-- Score text -->
        <text 
          x="${size/2}" 
          y="${size/2 - 5}" 
          text-anchor="middle" 
          dy="10" 
          class="progress-circle-score"
          fill="${color}"
        >
          ${score}
        </text>
        <text 
          x="${size/2}" 
          y="${size/2 + 30}" 
          text-anchor="middle" 
          class="progress-circle-label"
          fill="#999"
        >
          ${label}
        </text>
      </svg>
    </div>
  `;

  // Animate on load
  setTimeout(() => {
    const circle = container.querySelector('.progress-circle-fill');
    circle.style.strokeDasharray = `${progress} ${circumference}`;
  }, 100);
}

// ========================================
// ANIMATED GAUGE CHART
// ========================================
function createGaugeChart(containerId, percentage, label = '', colors = { low: '#28a745', medium: '#ffc107', high: '#dc3545' }) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const size = 200;
  const strokeWidth = 20;
  const radius = (size - strokeWidth) / 2;
  const circumference = Math.PI * radius; // Half circle
  const progress = (percentage / 100) * circumference;

  // Determine color based on percentage
  let color = colors.low;
  if (percentage > 66) color = colors.high;
  else if (percentage > 33) color = colors.medium;

  container.innerHTML = `
    <div class="gauge-chart-wrapper">
      <svg width="${size}" height="${size/2 + 40}" class="gauge-chart-svg">
        <!-- Background arc -->
        <path
          d="M ${strokeWidth/2} ${size/2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth/2} ${size/2}"
          fill="none"
          stroke="#e0e0e0"
          stroke-width="${strokeWidth}"
          stroke-linecap="round"
        />
        <!-- Progress arc -->
        <path
          class="gauge-chart-fill"
          d="M ${strokeWidth/2} ${size/2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth/2} ${size/2}"
          fill="none"
          stroke="${color}"
          stroke-width="${strokeWidth}"
          stroke-linecap="round"
          stroke-dasharray="0 ${circumference}"
        />
        <!-- Percentage text -->
        <text 
          x="${size/2}" 
          y="${size/2 + 10}" 
          text-anchor="middle" 
          class="gauge-chart-percentage"
          fill="${color}"
        >
          ${percentage}%
        </text>
        <text 
          x="${size/2}" 
          y="${size/2 + 35}" 
          text-anchor="middle" 
          class="gauge-chart-label"
          fill="#666"
        >
          ${label}
        </text>
      </svg>
    </div>
  `;

  // Animate on load
  setTimeout(() => {
    const arc = container.querySelector('.gauge-chart-fill');
    arc.style.strokeDasharray = `${progress} ${circumference}`;
  }, 100);
}

// ========================================
// ANIMATED COUNTER
// ========================================
function animateCounter(elementId, targetValue, duration = 2000, suffix = '') {
  const element = document.getElementById(elementId);
  if (!element) return;

  let startValue = 0;
  const startTime = Date.now();
  const isDecimal = targetValue % 1 !== 0;

  function updateCounter() {
    const elapsed = Date.now() - startTime;
    const progress = Math.min(elapsed / duration, 1);
    
    // Easing function (ease out cubic)
    const eased = 1 - Math.pow(1 - progress, 3);
    const currentValue = startValue + (targetValue - startValue) * eased;
    
    element.textContent = (isDecimal ? currentValue.toFixed(1) : Math.floor(currentValue)) + suffix;
    
    if (progress < 1) {
      requestAnimationFrame(updateCounter);
    } else {
      element.textContent = targetValue + suffix;
    }
  }

  updateCounter();
}

// ========================================
// HEALTH METRIC BAR
// ========================================
function createHealthMetricBar(containerId, value, maxValue, label, color = '#667eea') {
  const container = document.getElementById(containerId);
  if (!container) return;

  const percentage = (value / maxValue) * 100;

  container.innerHTML = `
    <div class="health-metric-bar">
      <div class="health-metric-header">
        <span class="health-metric-label">${label}</span>
        <span class="health-metric-value">${value} / ${maxValue}</span>
      </div>
      <div class="health-metric-track">
        <div class="health-metric-fill" style="background: ${color}; width: 0%;" data-width="${percentage}%"></div>
      </div>
    </div>
  `;

  // Animate bar
  setTimeout(() => {
    const fill = container.querySelector('.health-metric-fill');
    fill.style.width = fill.getAttribute('data-width');
  }, 100);
}

// ========================================
// RISK LEVEL INDICATOR
// ========================================
function createRiskIndicator(containerId, riskLevel) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const levels = ['Low', 'Moderate', 'High', 'Critical'];
  const colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545'];
  const currentIndex = levels.indexOf(riskLevel);

  let html = '<div class="risk-indicator">';
  levels.forEach((level, index) => {
    const isActive = index <= currentIndex;
    const color = isActive ? colors[index] : '#e0e0e0';
    html += `
      <div class="risk-level ${isActive ? 'active' : ''}" style="background: ${color};">
        <div class="risk-level-label">${level}</div>
      </div>
    `;
  });
  html += '</div>';

  container.innerHTML = html;
}

// Initialize all charts on page load
document.addEventListener('DOMContentLoaded', () => {
  // Auto-initialize any elements with data attributes
  document.querySelectorAll('[data-progress-circle]').forEach(el => {
    const score = parseInt(el.getAttribute('data-score'));
    const color = el.getAttribute('data-color') || '#667eea';
    const label = el.getAttribute('data-label') || 'out of 100';
    createProgressCircle(el.id, score, color, label);
  });

  document.querySelectorAll('[data-gauge-chart]').forEach(el => {
    const percentage = parseInt(el.getAttribute('data-percentage'));
    const label = el.getAttribute('data-label') || '';
    createGaugeChart(el.id, percentage, label);
  });

  document.querySelectorAll('[data-counter]').forEach(el => {
    const target = parseFloat(el.getAttribute('data-target'));
    const suffix = el.getAttribute('data-suffix') || '';
    animateCounter(el.id, target, 2000, suffix);
  });
});
