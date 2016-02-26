histomicstk.views.Visualization = girder.View.extend({
    initialize: function () {
        this.render();
        this._onresize = _.bind(this.render, this);
        $(window).on('resize', this._onresize);
    },
    render: _.debounce(function () {
        var width = this.$el.width(),
            height = this.$el.height();
        this.$el.html(histomicstk.templates.visualization({
            url: 'http://lorempixel.com/' + width + '/' + height + '/cats/'
        }));
    }, 500),
    destroy: function () {
        $(window).off('resize', this._onresize);
        girder.View.prototype.destroy.apply(this, arguments);
    }
});
