import Panel from 'girder_plugins/slicer_cli_web/views/Panel';

import annotationSelectorWidget from '../templates/annotationSelectorWidget.pug';
import '../stylesheets/annotationSelectorWidget.styl';

var AnnotationSelectorWidget = Panel.extend({
    events: _.extend(Panel.prototype.events, {
        'click .h-annotation > span': 'toggleAnnotation'
    }),
    initialize: function () {
        this.listenTo(girder.eventStream, 'g:event.job_status', function (evt) {
            if (evt.data.status > 2) {
                this.collection.fetch();
            }
        });
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
            this.listenTo(this.collection, 'add', this.render);
            this.collection.fetch();
        } else {
            this.collection = new Backbone.Collection();
        }
        return this;
    },
    render: function () {
        this.$el.html(annotationSelectorWidget({
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

export default AnnotationSelectorWidget;
