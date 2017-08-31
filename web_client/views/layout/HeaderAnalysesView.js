// import _ from 'underscore';
// import { restRequest } from 'girder/rest';

import events from '../../events';
// import router from '../../router';
import View from '../View';
import openAnalysis from '../../dialogs/openAnalysis';

import headerAnalysesTemplate from '../../templates/layout/headerAnalyses.pug';
// import '../../stylesheets/layout/headerAnalyses.styl';

var HeaderAnalysisView = View.extend({
    events: {
        'click .h-open-task': '_openDialog'
    },
    initialize() {
        this.image = null;
        this.listenTo(events, 'h:imageOpened', function (image) {
            this.image = image;
            this.render();
        });
    },
    render() {
        this.$el.html(
            headerAnalysesTemplate()
        );
        if (this.image) {
            this.$el.removeClass('hidden');
        } else {
            this.$el.addClass('hidden');
        }
        return this;
    },
    _openDialog(evt) {
        evt.preventDefault();
        openAnalysis();
    }
});

export default HeaderAnalysisView;
