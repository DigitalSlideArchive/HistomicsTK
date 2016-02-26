$(function () {
    histomicstk.events.trigger('g:appload.before');
    histomicstk.mainApp = new histomicstk.App({
        el: 'body',
        parentView: null
    });
    histomicstk.events.trigger('g:appload.after');
});

girder.router.enabled(false);
