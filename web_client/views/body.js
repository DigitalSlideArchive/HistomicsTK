import View from 'girder/views/View';
import PanelGroup from 'girder_plugins/slicer_cli_web/views/PanelGroup';

import Visualization from './visualization';
import events from '../events';
import * as dialogs from '../dialogs';

import body from '../templates/body.pug';
import '../stylesheets/body.styl';

var Body = View.extend({
    initialize: function () {
        this.visView = new Visualization({
            parentView: this
        });
        this.panelGroupView = new PanelGroup({
            parentView: this
        });
        this.listenTo(events, 'query:analysis', function (analysis) {
            if (analysis) {
                this.panelGroupView.setAnalysis(analysis);
            } else {
                this.panelGroupView.reset();
            }
        });
        this.listenTo(
            dialogs.image.model,
            'change',
            function (control) {
                if (!control || !control.get('value') || !control.get('value').id) {
                    return;
                }
                this._setImage(control.get('value'));
            }
        );
        this._setImage(dialogs.image.model.get('value'));
    },
    render: function () {
        this.$el.html(body());
        this.visView.setElement(this.$('#h-vis-container')).render();
        this.panelGroupView.setElement(this.$('#h-panel-controls-container')).render();
    },
    /**
     * This loops through all of the models in the control panels and sets any with type
     * "image" to the currently displayed image item.  This is a little hacky.  Ideally,
     * the CLI would provide a special annotation for the main image being processed
     * (whatever that might mean).
     */
    _setImage: function (item) {
        if (!item || !item.get('largeImage')) {
            return;
        }
        var file = new girder.models.FileModel({_id: item.get('largeImage').fileId});
        file.once('g:fetched', function () {
            this.panelGroupView
                .models(undefined, function (m) { return m.get('type') === 'image'; })
                .forEach(function (m) {
                    m.set('value', file);
                });
        }, this).fetch();
    }
});

export default Body;
