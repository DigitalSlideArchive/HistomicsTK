histomicstk.views.AnnotationSelectorWidget = girder.View.extend({
    events: {
        'click .h-annotation > span': 'toggleAnnotation'
    },
    setItem: function (item) {
        if (this.collection) {
            this.stopListening(this.collection);
        }

        this.parentItem = item;
        this.collection = this.parentItem.annotations;

        if (this.collection) {
            this.collection.append = true;

            this.listenTo(this.collection, 'g:changed', this.render);
            this.collection.fetch();
        } else {
            this.collection = new Backbone.Collection();
        }
        return this;
    },
    render: function () {
        this.$el.html(histomicstk.templates.annotationSelectorWidget({
            annotations: this.collection.toArray(),
            id: 'annotation-panel-container',
            title: 'Annotations'
        }));
        return this;
    },
    toggleAnnotation: function (evt) {
        var id = $(evt.currentTarget).data('id');
        var model = this.collection.get(id);
        model.set('displayed', !model.get('displayed'));
        this.render();
    }
});
