histomicstk.views.Header = girder.views.LayoutHeaderUserView.extend({
    render: function () {
        this.$el.html(histomicstk.templates.header({
            user: girder.currentUser
        }));
        return this;
    }
});
