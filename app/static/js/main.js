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

  // Optional: Flash messages fade out (but NOT warning alerts)
  const flash = document.querySelector('.alert.alert-dismissible, .flash-message');
  if (flash) {
    setTimeout(() => {
      flash.classList.add('fade');
    }, 4000);
  }
});
