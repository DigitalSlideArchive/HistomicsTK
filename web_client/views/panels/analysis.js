import View from '../View';

import analysisTemplate from '../../templates/panels/analysis.pug';
import '../../stylesheets/panels/analysis.styl';

var AnalysisPanel = View.extend({
    events: {
        'click .dropdown-toggle': function (evt) {
            var $el = $(evt.currentTarget);
            this.fixPosition($el);
        }
    },
    render() {
        this.$el.html(analysisTemplate({
            analyses: {
                docker1: {
                    '0.1': {
                        cli1: {
                            run: '/sdkf/sdkfjsdl/run'
                        }
                    },
                    '1.0': {
                        cli2: {
                            run: '/sdkf/sdkfjsdl/run'
                        }
                    }
                },
                docker2: {
                    '2.0': {
                        cli3: {
                            run: '/sdkf/sdkfjsdl/run'
                        }
                    }
                }
            }
        }));
    }
});

export default AnalysisPanel;
