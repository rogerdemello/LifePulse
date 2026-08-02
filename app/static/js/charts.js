/* The one chart this app draws.
 *
 * There were four builders here: a progress circle, a health-metric bar, a
 * risk indicator and a counter. No template called any of them and no template
 * carried their data attributes, so they had been dead for some time -- along
 * with two `#667eea` fallbacks, a colour from a palette two designs ago. The
 * health-score dial that looks like the progress circle is inline in
 * result_health_score.html and never used this file.
 *
 * **Colour is never written into the SVG.** Both surviving and deleted
 * builders used to read the design tokens once, at draw time, and copy the
 * resulting hex into `stroke=` and `fill=` attributes. A value copied like
 * that stops being a token: it cannot follow a theme change, and -- the reason
 * this was found -- it cannot follow the print block either. Printing a heart
 * result from a machine in dark mode drew the arc and the percentage in
 * #ff6961, a salmon meant for a black background, on white paper, while the
 * heading two lines below it printed in the proper print red because that one
 * came from CSS.
 *
 * `currentColor` defers the lookup to paint time, and the tone class on the
 * wrapper decides which token `color` resolves to. The track and the caption
 * take theirs from a class for the same reason.
 *
 * The gauge takes its tone from the page, not from an arbitrary split. It used
 * to decide for itself: >66% red, >33% amber, else green. Those thresholds
 * mean nothing here. Heart disease is flagged for follow-up above roughly 9%,
 * so a 39.7% risk -- four times the population rate -- drew an amber arc
 * directly above a red badge reading "39.74% estimated risk". The chart was
 * contradicting the assessment beside it. The template knows whether the
 * result crossed the model's own threshold and says so with data-tone.
 */
function createGaugeChart(containerId, percentage, label = '', tone = 'neutral') {
  const container = document.getElementById(containerId);
  if (!container) return;

  const size = 200;
  const strokeWidth = 20;
  const radius = (size - strokeWidth) / 2;
  const circumference = Math.PI * radius; // Half circle
  const progress = (percentage / 100) * circumference;

  container.innerHTML = `
    <div class="gauge-chart-wrapper chart-tone is-${tone}">
      <svg width="${size}" height="${size / 2 + 40}"
        <!-- Background arc -->
        <path
          class="chart-track"
          d="M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}"
          fill="none"
          stroke-width="${strokeWidth}"
          stroke-linecap="round"
        />
        <!-- Progress arc -->
        <path
          class="gauge-chart-fill"
          d="M ${strokeWidth / 2} ${size / 2} A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}"
          fill="none"
          stroke="currentColor"
          stroke-width="${strokeWidth}"
          stroke-linecap="round"
          stroke-dasharray="0 ${circumference}"
        />
        <!-- Percentage text -->
        <text
          x="${size / 2}"
          y="${size / 2 + 10}"
          text-anchor="middle"
          class="gauge-chart-percentage"
          fill="currentColor"
        >
          ${percentage}%
        </text>
        <text
          x="${size / 2}"
          y="${size / 2 + 35}"
          text-anchor="middle"
          class="gauge-chart-label chart-caption"
        >
          ${label}
        </text>
      </svg>
    </div>
  `;

  // Animate on load. Honoured by prefers-reduced-motion through the global
  // transition-duration override in the stylesheet.
  setTimeout(() => {
    const arc = container.querySelector('.gauge-chart-fill');
    arc.style.strokeDasharray = `${progress} ${circumference}`;
  }, 100);
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-gauge-chart]').forEach(el => {
    const percentage = parseInt(el.getAttribute('data-percentage'));
    const label = el.getAttribute('data-label') || '';
    const tone = el.getAttribute('data-tone') || 'neutral';
    createGaugeChart(el.id, percentage, label, tone);
  });
});
