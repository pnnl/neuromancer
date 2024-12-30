/* After much searching and experimenting, Sphinx's Read-The-Docs theme
 * Doesn't let you configure the table of contents in the sidebar (neither
 * {{ toc }} nor {{ toctree }}) to expand the sections that are initially
 * collaped but show a "+" button icon for expansion.
 *
 * Section expansion is handled client-side by javascript, and when
 * any toc node is clicked, events managed by minified js implement the
 * collapse/expand/focus logic.
 *
 * We therefore need to interpose on this logic with javascript.
 */

$(document).ready(() => {

  const menu = document.querySelector(".wy-menu ul li:first-child")
  const localtoc = document.querySelector(".local-toc ul li")

  const config = { attributes: true, attributeFilter: ["class"] };

  const observer = new MutationObserver((mutationsList) => {
    mutationsList.forEach((mutation) => {
      if (!mutation.target.classList.contains("current")) {
        mutation.target.classList.add("current");
      }
    });
  });


  if (menu) {
    menu.classList.add("current");
    observer.observe(menu, config);
  }

  if (localtoc) {
    localtoc.classList.add("current");
    observer.observe(localtoc, config);
  }

});
