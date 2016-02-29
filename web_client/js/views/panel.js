histomicstk.views.Panel = girder.View.extend({
    events: {
        'show.bs.collapse': 'expand',
        'hide.bs.collapse': 'collapse'
    },
    initialize: function (settings) {
        this.spec = settings.spec;
    },
    render: function () {
        this.$el.html(histomicstk.templates.panel(this.spec));
    },
    expand: function (e) {
        $(e.currentTarget).find('.icon-down-open').attr('class', 'icon-up-open');
    },
    collapse: function (e) {
        $(e.currentTarget).find('.icon-up-open').attr('class', 'icon-down-open');
    }
});
