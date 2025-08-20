const _get_storagekey = function(path) {
    let segments = path.replace(/^\/+|\/+$/g, '').split('/');
    return "mdft:" + (segments[1] || '') + ":key";
};

const _get_password = function(key) {
    let password = localStorage.getItem(key);
    if (!password)
    {
        let urlParams = new URLSearchParams(window.location.search);
        if (urlParams.has('k')) {
            password = urlParams.get('k');
            localStorage.setItem(key, password);
        }
    }
    return password;
};

document.addEventListener('DOMContentLoaded', () => {

    let key = _get_storagekey(location.pathname);
    let password = _get_password(key);
    if (!password)
    {
        // show banner if no password is stored
        var element = document.querySelector(".accesskey-warning");
        if (element)
            element.classList.remove("d-none");
    }
});
