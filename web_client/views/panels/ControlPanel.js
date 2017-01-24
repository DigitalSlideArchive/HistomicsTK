import { restRequest } from 'girder/rest';

import View from '../View';

import events from '../../events';
import controlPanel from '../../templates/panels/controlPanel.pug';
import '../../stylesheets/panels/controlPanel.styl';

var ControlPanel = View.extend({
    initialize(settings) {
        settings = settings || {};
        this.analysis = settings.analysis;
        this.listenTo(events, 'query:analysis', this.openAnalysis);
    },
    render() {
        this.$el.html(controlPanel());
        if (this.analysis) {
            restRequest({
                path: this.analysis + '/xmlspec',
                dataType: 'xml'
            }).then((xml) => {
                console.log(xml);
            });
        }
    },
    openAnalysis(q) {
        this.analysis = q;
        this.render();
    }
});

export default ControlPanel;
