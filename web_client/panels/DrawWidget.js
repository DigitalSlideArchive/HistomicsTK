import _ from 'underscore';

import Panel from 'girder_plugins/slicer_cli_web/views/Panel';

import drawWidget from '../templates/panels/drawWidget.pug';
import '../stylesheets/panels/drawWidget.styl';

var DrawWidget = Panel.extend({
    events: _.extend(Panel.prototype.events, {
        'click .h-save-annotation': 'saveAnnotation',
        'click .h-edit-element': 'editElement',
        'click .h-delete-element': 'deleteElement'
    }),
    initialize() {
    },
    render() {
        this.$el.html(drawWidget({
            title: 'Draw'
        }));
        this.$('.s-panel-content').collapse({toggle: false});
        return this;
    },
    saveAnnotation(evt) {
        console.log(evt);
    },
    editElement(evt) {
        console.log(evt);
    },
    deleteElement(evt) {
        console.log(evt);
    }
});

export default DrawWidget;
