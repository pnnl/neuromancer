$(document).ready(() => {

  const scriptElement = document.createElement('script');

  // Set the source of the script
  scriptElement.src = '/_static/js/load_late.js';

  // Append the script element to the end of the body
  document.body.appendChild(scriptElement);

});
