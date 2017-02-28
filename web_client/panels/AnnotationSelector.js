import Backbone from 'backbone';
import _ from 'underscore';

import { restRequest } from 'girder/rest';
import eventStream from 'girder/utilities/EventStream';
import Panel from 'girder_plugins/slicer_cli_web/views/Panel';

import annotationSelectorWidget from '../templates/panels/annotationSelector.pug';
import '../stylesheets/panels/annotationSelector.styl';

var AnnotationSelector = Panel.extend({
    events: _.extend(Panel.prototype.events, {
        'click .h-annotation > span': 'toggleAnnotation'
    }),
    initialize(settings) {
        this.collection = new Backbone.Collection();
        this.listenTo(this.collection, 'all', this.render);
        this.listenTo(eventStream, 'g:event.job_status', function (evt) {
            if (this.collection && evt.data.status > 2) {
                this.collection.fetch();
            }
        });
        if (settings.parentItem) {
            this.setItem(settings.parentItem);
        }
    },
    setItem(item) {
        this.parentItem = item;
        if (!this.parentItem) {
            this.collection.reset();
            return;
        }
        return restRequest({
            path: 'annotation',
            data: {
                itemId: this.parentItem.id
            }
        }).then((annotations) => {
            this.collection.reset(annotations);
        });
    },
    render() {
        this.$el.html(annotationSelectorWidget({
            annotations: this.collection.toArray(),
            id: 'annotation-panel-container',
            title: 'Annotations'
        }));
        return this;
    },
    toggleAnnotation(evt) {
        var id = $(evt.currentTarget).data('id');
        var model = this.collection.get(id);
        model.set('displayed', !model.get('displayed'));
        this.render();
    }
});

export default AnnotationSelector;
