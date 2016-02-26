histomicstk.App = girder.App.extend({

    render: function () {
        this.$el.html(histomicstk.templates.layout());

        new histomicstk.views.LayoutHeaderView({
            el: this.$('#h-app-header-container'),
            parentView: this
        }).render();

        return this;
    }

});
