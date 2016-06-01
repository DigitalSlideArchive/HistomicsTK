histomicstk.views.Header = girder.views.LayoutHeaderUserView.extend({
    events: {
        'click #g-analysis-menu a': '_selectAnalysis'
    },
    initialize: function () {
        this.analyses = [];
        girder.restRequest({path: 'HistomicsTK'})
            .then(_.bind(function (data) {
                this.analyses = data;
                this.render();
            }, this));
    },
    render: function () {
        this.$el.html(histomicstk.templates.header({
            user: girder.currentUser,
            analyses: this.analyses
        }));
        return this;
    },
    _selectAnalysis: function (evt) {
        var name = $(evt.currentTarget).data('name');
        histomicstk.router.navigate(name, {trigger: true});
    }
});
