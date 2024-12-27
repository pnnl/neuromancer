
function expand_menus() {

  var menu = document.querySelector(".wy-menu ul li:first-child")
  if (menu) {
    menu.classList.add("current");
  }

  var localtoc = document.querySelector(".local-toc ul li")
  if (localtoc) {
    localtoc.classList.add("current");
  }

}

function click_handler(event) {
  setTimeout(() => {
    expand_menus();
    window.removeEventListener("click", click_handler);
    window.addEventListener("click", click_handler);
  }, 10);
}

$(document).ready(() => {
  expand_menus();
  window.addEventListener("click", click_handler);
});
