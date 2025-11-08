// main.js - LifePulse global JavaScript

document.addEventListener('DOMContentLoaded', () => {
  console.log("LifePulse loaded 🚀");

  // Smooth scroll to top when navigation links are clicked
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      document.querySelector(this.getAttribute('href'))?.scrollIntoView({
        behavior: 'smooth'
      });
    });
  });

  // Show loading overlay on form submission
  const forms = document.querySelectorAll('form[method="post"]');
  const loadingOverlay = document.getElementById('loadingOverlay');
  
  forms.forEach(form => {
    form.addEventListener('submit', function(e) {
      // Check if form is valid before showing loading
      if (form.checkValidity()) {
        loadingOverlay.classList.add('active');
      }
    });
  });

  // Optional: Flash messages fade out (but NOT warning alerts)
  const flash = document.querySelector('.alert.alert-dismissible, .flash-message');
  if (flash) {
    setTimeout(() => {
      flash.classList.add('fade');
    }, 4000);
  }
});
